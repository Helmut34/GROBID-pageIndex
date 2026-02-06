"""
GROBID client for parsing academic PDFs into structured TEI XML.

Expects a GROBID service running at the configured URL (default: http://localhost:8070).

To start GROBID via Docker on the host:
    docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.1
"""

import httpx
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_GROBID_URL = "http://localhost:8070"
PROCESS_FULLTEXT_ENDPOINT = "/api/processFulltextDocument"
ISALIVE_ENDPOINT = "/api/isalive"


def is_alive(grobid_url: str = DEFAULT_GROBID_URL, timeout: float = 5.0) -> bool:
    try:
        resp = httpx.get(f"{grobid_url}{ISALIVE_ENDPOINT}", timeout=timeout)
        return resp.status_code == 200
    except httpx.ConnectError:
        return False


def process_pdf(
    pdf_path: str,
    grobid_url: str = DEFAULT_GROBID_URL,
    timeout: float = 120.0,
    consolidate_header: str = "1",
    consolidate_citations: str = "1",
    include_raw_citations: str = "1",
    segment_sentences: str = "0",
    tei_coordinates: str = "ref",
) -> str:
    """
    Send a PDF to GROBID and return the TEI XML response as a string.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    url = f"{grobid_url}{PROCESS_FULLTEXT_ENDPOINT}"

    with open(pdf_path, "rb") as f:
        files = {"input": (pdf_path.name, f, "application/pdf")}
        data = {
            "consolidateHeader": consolidate_header,
            "consolidateCitations": consolidate_citations,
            "includeRawCitations": include_raw_citations,
            "segmentSentences": segment_sentences,
            "teiCoordinates": tei_coordinates,
        }

        logger.info(f"Sending {pdf_path.name} to GROBID at {url}")
        resp = httpx.post(url, files=files, data=data, timeout=timeout)

    if resp.status_code != 200:
        raise RuntimeError(
            f"GROBID returned status {resp.status_code}: {resp.text[:500]}"
        )

    return resp.text
