"""
Stage 1: Load raw fault-code manuals, SOPs, and stop-code references from
data/raw/ into a normalized intermediate format before chunking.

Expected raw formats: PDF manuals, CSV/Excel fault-code tables, plain text
SOPs. Public sources (PLC vendor manuals, fault-code references) and your
sanitized company docs should both land in data/raw/ — tag their source in
metadata so you can filter or compare later.

TODO (you):
  - Drop your source files into data/raw/{public,company}/
  - Fill in `parse_fault_code_table()` for CSV/Excel fault-code sheets
  - Fill in `parse_manual_pdf()` for PDF manuals (unstructured / pypdf)
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FaultDoc:
    """One unit of knowledge: a fault code, stop code, or SOP section."""
    source: str              # "public" or "company"
    doc_type: str             # "fault_code" | "manual_section" | "sop"
    code: str | None = None   # e.g. "E402" — None for prose SOPs
    title: str = ""
    content: str = ""
    metadata: dict = field(default_factory=dict)


def parse_fault_code_table(path: Path, source: str) -> list[FaultDoc]:
    """
    Parse a fault-code table (CSV/Excel) into FaultDoc objects.
    Keep code + description + remedy TOGETHER as one chunk — don't split
    a fault code from its remedy, that destroys retrieval quality.
    """
    # TODO: pandas.read_csv / read_excel, one FaultDoc per row
    raise NotImplementedError


def parse_manual_pdf(path: Path, source: str) -> list[FaultDoc]:
    """
    Parse a PDF manual into section-level FaultDoc chunks.
    Use `unstructured` for layout-aware parsing (tables, headers) rather
    than raw pypdf text extraction, which loses structure.
    """
    # TODO: unstructured.partition.pdf.partition_pdf(path)
    raise NotImplementedError


def load_all(raw_dir: Path) -> list[FaultDoc]:
    """Walk data/raw/{public,company}/ and parse everything found."""
    docs: list[FaultDoc] = []
    for source in ("public", "company"):
        source_dir = raw_dir / source
        if not source_dir.exists():
            continue
        for f in source_dir.glob("**/*"):
            if f.suffix.lower() in (".csv", ".xlsx"):
                docs.extend(parse_fault_code_table(f, source))
            elif f.suffix.lower() == ".pdf":
                docs.extend(parse_manual_pdf(f, source))
    return docs


if __name__ == "__main__":
    raw_dir = Path(__file__).resolve().parents[2] / "data" / "raw"
    docs = load_all(raw_dir)
    print(f"Loaded {len(docs)} documents from {raw_dir}")
