"""
Production guardrails for the fault-diagnosis pipeline. Three layers,
checked in order — cheap and deterministic checks run first, so obviously
bad input never reaches the (expensive, fallible) agent chain at all:

1. Input validation  — length, empty input, basic prompt-injection patterns
2. Scope check        — retrieval score threshold (existing out-of-scope guard)
3. Output validation  — does the response actually look like a real
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

# Patterns that suggest an attempt to override system instructions rather
# than ask a genuine fault-diagnosis question. Not exhaustive — this is a
# basic first line of defense, not a substitute for the scope/output checks
# that follow it.
INJECTION_PATTERNS = [
    r"ignore (all |the )?(previous|prior|above) instructions",
    r"you are now",
    r"system prompt",
    r"disregard (all |the )?(previous|prior) (instructions|rules)",
    r"act as (if|a)",
    r"new instructions:",
    r"reveal your (prompt|instructions|system message)",
]

# A real diagnosis response should contain most of these structural
# markers. If a response is missing nearly all of them, something went
# wrong upstream (malformed generation, agent went off-script) and it
# shouldn't be shown as-is.
EXPECTED_OUTPUT_MARKERS = [
    "likely cause",
    "confidence",
    "remedy",
    "escalate",
]


def validate_input(query: str) -> tuple[bool, str]:
    """Returns (is_valid, reason_if_not). Cheap, deterministic, runs first."""
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
    """Returns (is_valid, reason_if_not). Sanity-checks the generated
    response actually looks like a real structured diagnosis before it
    goes anywhere near a user."""
    if not response or not response.strip():
        return False, "empty response"

    response_lower = response.lower()
    markers_present = sum(1 for m in EXPECTED_OUTPUT_MARKERS if m in response_lower)
    if markers_present < 2:
        return False, f"response missing expected structure ({markers_present}/4 markers found)"

    return True, ""


def safe_diagnose(query: str, diagnose_fn) -> str:
    """
    Wraps a diagnose function (e.g. crew.diagnose) with the full guardrail
    stack: input validation -> diagnose_fn (which does its own scope check)
    -> output validation -> catch-all error handling.
    """
    is_valid, reason = validate_input(query)
    if not is_valid:
        print(f"[guardrail] input rejected: {reason}")
        return SAFE_FALLBACK_MESSAGE

    try:
        response = diagnose_fn(query)
    except Exception as e:
        print(f"[guardrail] diagnose_fn raised an exception: {e}")
        return SAFE_FALLBACK_MESSAGE

    # The out-of-scope message from crew.py's own scope check is a valid,
    # intentional response — let it through without requiring it to match
    # the "structured diagnosis" output markers.
    if "doesn't appear to be about a PLC/SCADA fault" in response:
        return response

    is_valid_output, reason = validate_output(response)
    if not is_valid_output:
        print(f"[guardrail] output rejected: {reason}")
        return SAFE_FALLBACK_MESSAGE

    return response