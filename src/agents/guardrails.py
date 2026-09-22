"""
Production guardrails for the fault-diagnosis pipeline. Layers, checked in
order — cheap and deterministic checks run first, so obviously bad input
never reaches the (expensive, fallible) agent chain at all:

1. Input validation   — length, empty input, basic prompt-injection patterns
2. Scope check         — retrieval score threshold (in crew.py's diagnose())
3. Response sanitization — strips leaked CrewAI prompt-template scaffolding
                           (see clean_response() docstring)
4. Output validation   — does the response actually look like a real
                          diagnosis, not something malformed or off-template

Anything that fails any layer returns the same clean, generic message —
deliberately generic, so failure reasons (which could hint at system
internals) are never exposed to whoever is asking.
"""

import re

SAFE_FALLBACK_MESSAGE = (
    "I'm not able to process that request. Please enter a specific PLC/SCADA "
    "stop code, error code, or fault description."
)

MAX_QUERY_LENGTH = 500
MIN_QUERY_LENGTH = 3

INJECTION_PATTERNS = [
    r"ignore (all |the )?(previous|prior|above) instructions",
    r"you are now",
    r"system prompt",
    r"disregard (all |the )?(previous|prior) (instructions|rules)",
    r"act as (if|a)",
    r"new instructions:",
    r"reveal your (prompt|instructions|system message)",
]

EXPECTED_OUTPUT_MARKERS = [
    "likely cause",
    "confidence",
    "remedy",
    "escalate",
]

# CrewAI's own internal task-prompt template includes phrases like these.
# A weaker/quantized local model sometimes echoes fragments of its received
# prompt back into its generated answer instead of cleanly separating
# "instructions I was given" from "the answer I should produce" — this is
# a known small-model failure mode, not a bug in our prompts. These markers
# are the LAST piece of that boilerplate CrewAI appends before the model's
# actual answer, so anything up to and including the last occurrence of one
# of these is framework scaffolding, not real content.
CREWAI_BOILERPLATE_MARKERS = [
    r"your job depends on it!",
    r"begin!\s*this is very important",
    r"you must return the actual complete content as the final answer",
]


def clean_response(response: str) -> str:
    """Strips leaked CrewAI prompt-template text that a weaker local model
    sometimes echoes back. Finds the LAST occurrence of any known
    boilerplate marker and keeps only what comes after it — CrewAI's
    template always precedes the model's real answer, never follows it."""
    if not response:
        return response

    last_cut = -1
    for pattern in CREWAI_BOILERPLATE_MARKERS:
        for m in re.finditer(pattern, response, re.IGNORECASE):
            last_cut = max(last_cut, m.end())

    cleaned = response[last_cut:].strip() if last_cut != -1 else response

    # Cosmetic: drop a leading "Final Diagnosis:" / "Final Answer:" label
    # some models prepend after the scaffolding is stripped.
    cleaned = re.sub(r"^(final (diagnosis|answer):?\s*)", "", cleaned, flags=re.IGNORECASE).strip()

    return cleaned or response  # never return an empty string from cleaning


def validate_input(query: str) -> tuple[bool, str]:
    if not query or not query.strip():
        return False, "empty query"
    query_stripped = query.strip()
    if len(query_stripped) < MIN_QUERY_LENGTH:
        return False, "query too short"
    if len(query_stripped) > MAX_QUERY_LENGTH:
        return False, "query too long"
    query_lower = query_stripped.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, query_lower):
            return False, "possible instruction-override attempt"
    return True, ""


def validate_output(response: str) -> tuple[bool, str]:
    """Structural check, not keyword-counting: the response must actually
    match our fixed template (see crew.py's format_diagnosis), meaning it
    has real, non-empty content under at least the Likely Cause and Remedy
    Steps headers. A response missing these isn't "close enough" — it means
    JSON parsing failed upstream or the model returned something malformed,
    and that should fall back to the safe message rather than be shown."""
    if not response or not response.strip():
        return False, "empty response"

    cause_match = re.search(r"1\. Likely Cause\s*\n\s*(.+)", response)
    remedy_match = re.search(r"3\. Remedy Steps\s*\n(.+?)(?:\n\n|$)", response, re.DOTALL)

    if not cause_match or not cause_match.group(1).strip() or cause_match.group(1).strip().lower() == "not specified":
        return False, "missing or empty likely cause"
    if not remedy_match or not remedy_match.group(1).strip():
        return False, "missing remedy steps"

    return True, ""


def safe_diagnose(query: str, diagnose_fn) -> str:
    """Full guardrail stack: input validation -> diagnose_fn -> response
    sanitization -> output validation -> catch-all error handling."""
    is_valid, reason = validate_input(query)
    if not is_valid:
        print(f"[guardrail] input rejected: {reason}")
        return SAFE_FALLBACK_MESSAGE

    try:
        response = diagnose_fn(query)
    except Exception as e:
        print(f"[guardrail] diagnose_fn raised an exception: {e}")
        return SAFE_FALLBACK_MESSAGE

    if "doesn't appear to be about a PLC/SCADA fault" in response:
        return response

    response = clean_response(response)

    is_valid_output, reason = validate_output(response)
    if not is_valid_output:
        print(f"[guardrail] output rejected: {reason}")
        return SAFE_FALLBACK_MESSAGE

    return response