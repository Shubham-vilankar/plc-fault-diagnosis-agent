import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pdfplumber


"""
Stage 1: Loaded raw fault-code manuals, SOPs, and stop-code references from
data/raw/ into a normalized intermediate format before chunking.
"""
@dataclass
class FaultDoc:
    """One unit of knowledge: ie- a fault code, stop code, or SOP section."""
    source: str               # "public" or "company"
    doc_type: str              # "fault_code" | "manual_section" | "sop"
    code: str | None = None    # e.g. "E402" — None for prose SOPs
    title: str = ""
    content: str = ""
    confidence: str = "high"   # "high" (structured table) | "low" (regex guess)
    metadata: dict = field(default_factory=dict)


# Matches common PLC fault/error code tokens:
#   hex pairs like "00F0 0001", single hex "0010", or vendor-style "E402" / "F001"
CODE_PATTERN = re.compile(
    r"^\s*((?:[0-9A-F]{2,4}\s+){1,2}[0-9A-F]{2,4}|[A-Z]\d{2,4})\s+(.+)$"
)


def parse_fault_code_table(path: Path, source: str) -> list[FaultDoc]:
    """
    This fucntion Parse a fault-code table (CSV/Excel) into FaultDoc objects. Every row
    becomes one FaultDoc — code + description + remedy stay together as a
    single chunk .
    """
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    df.columns = [str(c).strip().lower() for c in df.columns]

    code_col = next((c for c in df.columns if "err" in c or "code" in c or "fault" in c), None)
    desc_col = next((c for c in df.columns if "desc" in c or "message" in c), None)
    remedy_col = next((c for c in df.columns if "remedy" in c or "action" in c or "fix" in c), None)

    if code_col is None or desc_col is None:
        warnings.warn(
            f"{path.name}: couldn't confidently identify code/description "
            f"columns from {list(df.columns)} — check this file manually."
        )
        code_col = code_col or df.columns[0]
        desc_col = desc_col or df.columns[1]

    docs = []
    for _, row in df.iterrows():
        code = str(row.get(code_col, "")).strip()
        desc = str(row.get(desc_col, "")).strip()
        remedy = str(row.get(remedy_col, "")).strip() if remedy_col else ""
        if not code or code.lower() == "nan":
            continue
        content = f"Fault code {code}: {desc}"
        if remedy and remedy.lower() != "nan":
            content += f"\nRemedy: {remedy}"
        docs.append(FaultDoc(
            source=source, doc_type="fault_code", code=code,
            title=f"Fault {code}", content=content, confidence="high",
            metadata={"file": path.name},
        ))
    return docs


CID_ARTIFACT = re.compile(r"\(cid:\d+\)")


def _clean_desc(desc: str) -> str:
    """Strip PDF font-encoding artifacts (unmapped glyphs show up as
    '(cid:129)' etc.) and collapse whitespace/newlines from wrapped cells."""
    desc = CID_ARTIFACT.sub("", desc)
    return re.sub(r"\s+", " ", desc).strip()


def _looks_like_code(token: str) -> bool:
    """A real fault/error code always contains at least one digit
    ('00F0', '101', 'D8099'). Without this check, English words made
    entirely of hex letters (a-f) — 'Add', 'Cab', 'Dead', 'Face' — false-
    positive match the hex pattern and pull in unrelated tables (e.g. an
    instruction-mnemonic table instead of a fault-code table)."""
    return bool(re.match(r"^[0-9A-Fa-fXx]{2,6}$", token)) and bool(re.search(r"\d", token))


def _looks_like_prose(desc: str, min_words: int = 3) -> bool:
    """A real fault description has several actual words ('Configuration
    error', 'Communication module not working') — not just one word next to
    a pile of hex/byte tokens
    """
    real_words = re.findall(r"[A-Za-z]{3,}", desc)
    return len(real_words) >= min_words


def _extract_table_rows(pdf_path: Path) -> list[FaultDoc]:
    """this will extract table rows"""
    docs = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            for table in page.extract_tables():
                for row in table:
                    if not row or len(row) < 2:
                        continue
                    cells = [c.strip() if c else "" for c in row]
                    
                    if cells[0] and _looks_like_code(cells[0]):
                        desc = _clean_desc(" ".join(cells[1:]))
                        if desc and _looks_like_prose(desc):
                            docs.append(FaultDoc(
                                source="public", doc_type="fault_code",
                                code=cells[0], title=f"Fault {cells[0]}",
                                content=f"Fault code {cells[0]}: {desc}",
                                confidence="high",
                                metadata={"file": pdf_path.name, "page": page_num + 1},
                            ))
    return docs


def _extract_regex_fallback(pdf_path: Path) -> list[FaultDoc]:
    docs = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            for line in text.splitlines():
                m = CODE_PATTERN.match(line)
                if m:
                    code, desc = m.group(1).strip(), _clean_desc(m.group(2))
                    if len(desc) > 10 and _looks_like_prose(desc, min_words=2):
                        docs.append(FaultDoc(
                            source="public", doc_type="fault_code",
                            code=code, title=f"Fault {code}",
                            content=f"Fault code {code}: {desc}",
                            confidence="low",
                            metadata={"file": pdf_path.name, "page": page_num + 1},
                        ))
    return docs


def parse_manual_pdf(path: Path, source: str) -> list[FaultDoc]:
    """
    this function parses a PDF manual into fault-code chunks. Tries structured table
    extraction first (works for manuals with real tables, e.g. the
    Allen-Bradley quick reference) (e.g. Siemens alarm descriptions, Mitsubishi
    programming manual) and marks those chunks low-confidence.

    Low-confidence chunks aren't discarded — they still go into the
    knowledge base 
    """
    table_docs = _extract_table_rows(path)
    if table_docs:
        for d in table_docs:
            d.source = source
        return table_docs

    fallback_docs = _extract_regex_fallback(path)
    for d in fallback_docs:
        d.source = source
    return fallback_docs


def load_all(raw_dir: Path) -> list[FaultDoc]:
    """Load all into  data/raw/{public,company}/ and parse everything found."""
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
    from collections import Counter

    raw_dir = Path(__file__).resolve().parents[2] / "data" / "raw"
    docs = load_all(raw_dir)
    high = sum(1 for d in docs if d.confidence == "high")
    low = sum(1 for d in docs if d.confidence == "low")
    print(f"Loaded {len(docs)} fault-code chunks from {raw_dir}")
    print(f"  {high} high-confidence (table-extracted), {low} low-confidence (regex fallback — review these)")

    by_file = Counter(d.metadata.get("file", "?") for d in docs)
    print("\nBreakdown by source file:")
    for fname, count in by_file.most_common():
        print(f"  {fname}: {count}")

    print("\nSample chunks (up to 2 per file):")
    seen_per_file: dict[str, int] = {}
    for d in docs:
        fname = d.metadata.get("file", "?")
        if seen_per_file.get(fname, 0) < 2:
            print(f"  [{fname}] {d.content!r}")
            seen_per_file[fname] = seen_per_file.get(fname, 0) + 1