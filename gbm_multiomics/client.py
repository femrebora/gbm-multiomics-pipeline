"""
client.py — Generic GDC REST API client for GBM multiomics downloads.

All data types use the same two-phase approach:
  1. POST /files  → discover file UUIDs matching filter criteria
  2. POST /data   → stream-download a ZIP archive by UUID list

This client is intentionally independent of tcga-gdc-downloader internals
so the two packages can evolve separately.
"""

from __future__ import annotations

import time
from pathlib import Path

import requests

from gbm_multiomics.constants import (
    GDC_FILES_ENDPOINT, GDC_CASES_ENDPOINT, GDC_DATA_ENDPOINT,
    GDC_STATUS_ENDPOINT, GDC_PROJECTS_ENDPOINT,
    BASE_FILE_FIELDS, CLINICAL_FIELDS,
    GDC_PAGE_SIZE, MAX_TOTAL_FILES,
    REQUEST_TIMEOUT_SHORT, REQUEST_TIMEOUT_LONG,
    MAX_RETRIES, RETRY_WAIT_SECONDS, CHUNK_SIZE_BYTES,
    GDC_DOWNLOAD_BATCH_SIZE,
)


class GDCError(Exception):
    """Structured error from the GDC API or download pipeline."""

    def __init__(self, message: str, fix: str = "", step: str = ""):
        super().__init__(message)
        self.message = message
        self.fix = fix
        self.step = step

    def formatted(self) -> str:
        lines = [f"  GDC Error [{self.step}]: {self.message}"]
        if self.fix:
            lines.append(f"  Fix: {self.fix}")
        return "\n".join(lines)


class GBMClient:
    """
    GDC API client for TCGA-GBM multiomics data download.

    Usage
    -----
    client = GBMClient()                           # open-access
    client = GBMClient(token="your-token-string")  # controlled-access
    client = GBMClient.from_file("/path/to/token") # token from file
    """

    def __init__(self, token: str | None = None):
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        if token:
            self._session.headers.update({"X-Auth-Token": token.strip()})

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_file(cls, token_path: str) -> "GBMClient":
        """Load a GDC auth token from a text file."""
        p = Path(token_path)
        if not p.exists():
            raise GDCError(
                f"Token file not found: {token_path}",
                fix="Download a token at https://portal.gdc.cancer.gov (Login → Download Token)",
                step="auth",
            )
        token = p.read_text(encoding="utf-8").strip()
        if len(token) < 20:
            raise GDCError("Token file appears empty.", step="auth",
                           fix="Download a fresh token from https://portal.gdc.cancer.gov")
        return cls(token=token)

    # ── Low-level HTTP ────────────────────────────────────────────────────────

    def _post(
        self,
        endpoint: str,
        payload: dict,
        context: str,
        stream: bool = False,
        timeout: int = REQUEST_TIMEOUT_SHORT,
    ) -> requests.Response:
        """POST with exponential-backoff retry. Translates HTTP errors to GDCError."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self._session.post(
                    endpoint, json=payload, stream=stream, timeout=timeout
                )
                resp.raise_for_status()
                return resp

            except requests.exceptions.Timeout:
                if attempt < MAX_RETRIES:
                    print(f"  ⚠  Timeout (attempt {attempt}/{MAX_RETRIES}), "
                          f"retrying in {RETRY_WAIT_SECONDS}s...")
                    time.sleep(RETRY_WAIT_SECONDS)
                else:
                    raise GDCError(f"Request timed out: {context}",
                                   fix="Check your connection or retry later.",
                                   step=context)

            except requests.exceptions.ConnectionError:
                raise GDCError(f"Cannot connect to GDC: {context}",
                               fix="Check internet access or https://status.gdc.cancer.gov",
                               step=context)

            except requests.exceptions.HTTPError:
                status = resp.status_code
                body = ""
                try:
                    body = resp.text[:300]
                except Exception:
                    pass
                if status == 401:
                    raise GDCError("Authentication failed (401).",
                                   fix="Check your GDC token.", step=context)
                if status == 403:
                    raise GDCError("Access denied (403). Controlled-access data requires a token.",
                                   fix="Download a token at https://portal.gdc.cancer.gov",
                                   step=context)
                if status == 429:
                    if attempt < MAX_RETRIES:
                        print("  ⚠  Rate limited. Waiting 60s before retry...")
                        time.sleep(60)
                        continue
                    raise GDCError("Rate limited (429).", step=context)
                raise GDCError(f"HTTP {status}: {body}", step=context)

        raise GDCError(f"Failed after {MAX_RETRIES} attempts: {context}", step=context)

    # ── Connectivity ──────────────────────────────────────────────────────────

    def check_connectivity(self) -> bool:
        """Return True if the GDC API is reachable."""
        try:
            return self._session.get(GDC_STATUS_ENDPOINT, timeout=10).status_code == 200
        except Exception:
            return False

    # ── File discovery ────────────────────────────────────────────────────────

    def discover_files(
        self,
        project_id: str,
        data_type_filters: list[dict],
        extra_fields: str = "",
    ) -> list[dict]:
        """
        Return file metadata for all open-access files matching the given filters.

        Parameters
        ----------
        project_id : str
            e.g. "TCGA-GBM"
        data_type_filters : list[dict]
            GDC filter objects for the desired data type (from constants.py).
        extra_fields : str
            Additional comma-separated fields to request beyond BASE_FILE_FIELDS.

        Returns
        -------
        list[dict]
            Each dict has at minimum: file_id, file_name, file_size, cases[...]
        """
        project_filter = {
            "op": "=",
            "content": {"field": "cases.project.project_id", "value": project_id},
        }
        combined = {"op": "and", "content": [project_filter] + data_type_filters}
        fields = BASE_FILE_FIELDS
        if extra_fields:
            fields = f"{fields},{extra_fields}"

        payload = {
            "filters": combined,
            "fields": fields,
            "size": GDC_PAGE_SIZE,
            "from": 0,
            "format": "json",
        }

        all_hits: list[dict] = []
        page = 0
        while True:
            payload["from"] = page * GDC_PAGE_SIZE
            resp = self._post(GDC_FILES_ENDPOINT, payload, context="file discovery")
            data = resp.json().get("data", {})
            hits = data.get("hits", [])
            total = data.get("pagination", {}).get("total", len(hits))
            all_hits.extend(hits)

            if len(all_hits) >= total or len(all_hits) >= MAX_TOTAL_FILES:
                break
            page += 1
            print(f"  📄  Retrieved {len(all_hits)}/{total} records...")

        if not all_hits:
            raise GDCError(
                f"No files found for project '{project_id}' with the given filters.",
                fix=(
                    "1. Run `gbm-download --data-type <type> --dry-run` to check.\n"
                    "2. Verify the project has this data type at https://portal.gdc.cancer.gov"
                ),
                step="file discovery",
            )
        return all_hits

    # ── Clinical data ─────────────────────────────────────────────────────────

    def fetch_clinical_data(self, project_id: str) -> list[dict]:
        """Return raw clinical records for all cases in the project."""
        resp = self._post(
            GDC_CASES_ENDPOINT,
            {
                "filters": {"op": "=", "content": {
                    "field": "project.project_id", "value": project_id,
                }},
                "fields": CLINICAL_FIELDS,
                "size": GDC_PAGE_SIZE,
                "format": "json",
            },
            context="clinical data fetch",
        )
        return resp.json().get("data", {}).get("hits", [])

    # ── Batch download ────────────────────────────────────────────────────────

    def batch_download(
        self,
        file_ids: list[str],
        dest_dir: Path,
        label: str = "data",
    ) -> list[Path]:
        """
        Download file_ids in batches using POST /data, extract each archive.

        Returns
        -------
        list[Path]
            Directories extracted from each batch archive.
        """
        import zipfile, tarfile

        dest_dir.mkdir(parents=True, exist_ok=True)
        n_batches = (len(file_ids) + GDC_DOWNLOAD_BATCH_SIZE - 1) // GDC_DOWNLOAD_BATCH_SIZE
        print(f"  📦  Downloading {len(file_ids)} {label} files "
              f"in {n_batches} batch{'es' if n_batches > 1 else ''}...")

        extracted_dirs: list[Path] = []

        for batch_num, start in enumerate(
            range(0, len(file_ids), GDC_DOWNLOAD_BATCH_SIZE), start=1
        ):
            batch_ids  = file_ids[start:start + GDC_DOWNLOAD_BATCH_SIZE]
            batch_path = dest_dir.parent / f"{label}_batch_{batch_num:03d}.zip"

            if not batch_path.exists():
                print(f"  ⬇   Batch {batch_num}/{n_batches} ({len(batch_ids)} files)...")
                self._stream_to_file(batch_ids, batch_path)
            else:
                print(f"  ✅  Batch {batch_num}/{n_batches}: already downloaded.")

            print(f"  📂  Extracting batch {batch_num}/{n_batches}...")
            extracted_dirs.append(
                self._extract_archive(batch_path, dest_dir, batch_num, n_batches)
            )

        return extracted_dirs

    def _stream_to_file(self, file_ids: list[str], dest: Path) -> None:
        """Stream a GDC /data POST response to dest."""
        resp = self._post(
            GDC_DATA_ENDPOINT,
            {"ids": file_ids},
            context="bulk download",
            stream=True,
            timeout=REQUEST_TIMEOUT_LONG,
        )
        total_size = int(resp.headers.get("content-length", 0))
        written = 0

        try:
            from tqdm import tqdm
            pbar = tqdm(total=total_size or None, unit="B", unit_scale=True,
                        unit_divisor=1024, desc="  Downloading", ncols=70)
        except ImportError:
            pbar = _DummyProgress()

        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE_BYTES):
                if chunk:
                    fh.write(chunk)
                    written += len(chunk)
                    pbar.update(len(chunk))

        if hasattr(pbar, "close"):
            pbar.close()

    def _extract_archive(
        self, archive_path: Path, dest_dir: Path, batch_num: int, n_batches: int
    ) -> Path:
        """Detect ZIP vs tar.gz, extract to dest_dir, delete archive. Returns dest_dir."""
        import zipfile, tarfile

        with open(archive_path, "rb") as fh:
            magic = fh.read(4)

        try:
            if magic[:2] == b"\x1f\x8b":
                tgz = archive_path.with_suffix(".tar.gz")
                archive_path.rename(tgz)
                with tarfile.open(tgz, "r:gz") as t:
                    t.extractall(dest_dir)
                tgz.unlink(missing_ok=True)
            elif magic[:4] == b"PK\x03\x04":
                with zipfile.ZipFile(archive_path, "r") as z:
                    z.extractall(dest_dir)
                archive_path.unlink(missing_ok=True)
            else:
                raise GDCError(
                    f"Batch {batch_num} is not a ZIP or tar.gz "
                    f"(magic: {magic.hex()}, size: {archive_path.stat().st_size / 1024:.1f} KB).",
                    fix="Re-run — the download was likely truncated.",
                    step="extraction",
                )
        except (EOFError, tarfile.ReadError, zipfile.BadZipFile) as exc:
            for p in [archive_path, archive_path.with_suffix(".tar.gz")]:
                if p.exists():
                    p.unlink()
            raise GDCError(
                f"Batch {batch_num}/{n_batches} archive corrupted: {exc}",
                fix="Re-run — the corrupted batch file has been deleted and will be retried.",
                step="extraction",
            )
        return dest_dir

    # ── CDR helper ────────────────────────────────────────────────────────────

    def download_cdr(self, dest_dir: Path) -> Path:
        """Download the PanCanAtlas CDR Excel file and return its path."""
        from gbm_multiomics.constants import CDR_FILE_UUID, CDR_CACHE_FILENAME
        dest = dest_dir / CDR_CACHE_FILENAME
        if dest.exists():
            print(f"  ✅  CDR cache found: {dest.name}")
            return dest
        print("  📚  Downloading PanCanAtlas CDR...")
        self._stream_to_file([CDR_FILE_UUID], dest)
        return dest


class _DummyProgress:
    def update(self, n: int) -> None:
        pass

    def close(self) -> None:
        pass
