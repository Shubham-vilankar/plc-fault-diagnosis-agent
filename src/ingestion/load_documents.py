from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FaultDoc:
    """One unit of knowledge: ie- a fault code, stop code, or SOP section."""
    source: str              # "public" or "company"
    doc_type: str             # "fault_code" | "manual_section" | "sop"
    code: str | None = None   # e.g. "E402" — None for prose SOPs
    title: str = ""
    content: str = ""
    metadata: dict = field(default_factory=dict)


def parse_fault_code_table(path: Path, source: str) -> list[FaultDoc]:
    """
    this function would Parse a fault-code table (CSV/Excel) into FaultDoc objects.
    
    """
    raise NotImplementedError


def parse_manual_pdf(path: Path, source: str) -> list[FaultDoc]:
    """
    this function would Parse a PDF manual into section-level FaultDoc chunks.
    """
    raise NotImplementedError


def load_all(raw_dir: Path) -> list[FaultDoc]:
    """Load all data/raw/{public,company}/ and parse everything found."""
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
