"""
This is simple frontend UI for the PLC/SCADA fault-diagnosis agent, built with Streamlit.
Prerequisites: src/serve/local_llm_server.py must be running on port 8001

Run with: streamlit run src/frontend/app.py
"""

import re
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

st.set_page_config(page_title="PLC/SCADA Fault Diagnosis Agent", page_icon="🔧", layout="centered")

st.title("🔧 PLC/SCADA Fault-Diagnosis Agent")
st.caption(
    "Retrieval-augmented, fine-tuned, multi-agent diagnosis over real "
    "Siemens / Allen-Bradley / Mitsubishi fault-code documentation."
)

with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "Ask about a PLC/SCADA stop code, error code, or fault symptom. "
        "The agent retrieves relevant documentation, reasons through a "
        "diagnosis, and decides whether it should be escalated to a human "
        "technician.\n\n"
        "If the question isn't something the knowledge base actually "
        "covers, it will say so honestly rather than guess."
    )
    st.markdown("---")
    st.markdown("**Try:**")
    st.markdown(
        "- *Stop code E402 on Siemens S7-1200, motor won't start*\n"
        "- *Communication module not working, what should I check?*"
    )


@st.cache_resource(show_spinner=False)
def get_diagnose_fn():
    """Import lazily and cache — avoids re-importing (and re-initializing
    the embedder/Qdrant client used by the scope guard) on every rerun."""
    from agents.crew import diagnose
    from agents.guardrails import safe_diagnose
    return diagnose, safe_diagnose


def render_structured_response(response: str) -> None:
    """Splits the numbered response sections out for nicer display, falling
    back to plain text if the response doesn't match the expected format
    (e.g. the out-of-scope / safe-fallback messages, which are intentionally
    plain sentences, not structured diagnoses)."""
    parts = re.split(r"\n?\d\.\s+", response)
    parts = [p for p in parts if p.strip()]  # drop empty preamble before "1."

    if len(parts) < 2:
        # Not a structured diagnosis (out-of-scope message, fallback, etc.)
        st.info(response)
        return

    icons = {"likely cause": "🔍", "confidence": "📊", "remedy steps": "🛠️", "escalate": "🚨"}
    for part in parts:
        label, _, body = part.partition("\n")
        icon = next((v for k, v in icons.items() if k in label.lower()), "•")
        st.markdown(f"**{icon} {label.strip()}**")
        if body.strip():
            st.markdown(body.strip())
        st.markdown("")


query = st.text_input(
    "Describe the fault or enter a stop/error code:",
    placeholder="e.g. Stop code E402 on Siemens S7-1200, motor won't start",
)

if st.button("Diagnose", type="primary") and query:
    diagnose, safe_diagnose = get_diagnose_fn()
    with st.spinner("Retrieving relevant documentation and diagnosing..."):
        try:
            result = safe_diagnose(query, diagnose)
        except Exception as e:
            st.error(
                "Something went wrong reaching the model server. Make sure "
                "local_llm_server.py is running on port 8001."
            )
            st.exception(e)
            result = None

    if result:
        st.markdown("---")
        render_structured_response(result)

elif not query and st.session_state.get("_submitted"):
    st.warning("Please enter a fault description or code.")