
import sys
from pathlib import Path
from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM

"""
Stage 4: CrewAI agent layer — Retriever, Diagnostician, Escalator roles on
top of the fine-tuned model + RAG stack, replacing the single-shot
baseline_rag.py flow with a proper multi-step agent pipeline.

Points at the local FastAPI server (src/serve/local_llm_server.py) via
CrewAI's OpenAI-compatible LLM interface.

Make Sure the Python Server is running
(uvicorn src.serve.local_llm_server:app --port 8001)
"""

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from agents.tools import FaultCodeRetrieverTool  # noqa: E402

local_llm = LLM(
    model="openai/fault-diagnosis",
    base_url="http://localhost:8001/v1",
    api_key="not-needed",
    temperature=0.1,
)

retriever_tool = FaultCodeRetrieverTool()

retriever_agent = Agent(
    role="Fault Code Retriever",
    goal="Find the most relevant fault-code entries for a given PLC/SCADA query",
    backstory=(
        "You are a retrieval specialist for industrial fault diagnosis. "
        "You search the knowledge base and surface the most relevant fault "
        "code entries — you do not diagnose or interpret, only retrieve."
    ),
    tools=[retriever_tool],
    llm=local_llm,
    verbose=True,
)

diagnostician_agent = Agent(
    role="Fault Diagnostician",
    goal="Diagnose the likely cause and remedy for a PLC/SCADA fault based on retrieved reference material",
    backstory=(
        "You are an experienced industrial automation technician. Given "
        "retrieved fault-code reference material, you reason through the "
        "likely cause and remedy steps. You are honest about uncertainty — "
        "if the reference material doesn't clearly cover the query, you say "
        "so rather than guessing."
    ),
    llm=local_llm,
    verbose=True,
)

escalator_agent = Agent(
    role="Escalation Reviewer",
    goal="Review a proposed diagnosis and decide whether it should be escalated to a human technician",
    backstory=(
        "You are a safety-conscious reviewer. You look at a proposed "
        "diagnosis and its confidence level, and decide whether a human "
        "technician needs to be involved before acting on it. You escalate "
        "readily when confidence is low or the situation could be unsafe."
    ),
    llm=local_llm,
    verbose=True,
)


def build_crew(query: str) -> Crew:
    retrieve_task = Task(
        description=f"Search the fault-code knowledge base for entries relevant to: '{query}'",
        expected_output="A list of relevant fault-code entries with their descriptions and confidence levels.",
        agent=retriever_agent,
    )

    diagnose_task = Task(
        description=(
            f"Given the retrieved fault-code entries, diagnose the query: '{query}'. "
            "Respond in this structure:\n"
            "1. Likely Cause\n2. Confidence (low/medium/high)\n3. Remedy Steps\n\n"
            "If the retrieved material doesn't clearly cover the query, say so honestly."
        ),
        expected_output="A structured diagnosis with likely cause, confidence, and remedy steps.",
        agent=diagnostician_agent,
        context=[retrieve_task],
    )

    escalate_task = Task(
        description=(
            "Review the proposed diagnosis and its confidence level. Decide "
            "whether this should be escalated to a human technician, and "
            "append your decision with a brief reason to the final answer."
        ),
        expected_output=(
            "The final diagnosis, with an added '4. Escalate to a technician? "
            "(yes/no, with reason)' section."
        ),
        agent=escalator_agent,
        context=[diagnose_task],
    )

    return Crew(
        agents=[retriever_agent, diagnostician_agent, escalator_agent],
        tasks=[retrieve_task, diagnose_task, escalate_task],
        process=Process.sequential,
        verbose=True,
    )


if __name__ == "__main__":
    test_queries = [
        "Stop code E402 on Siemens S7-1200, motor won't start",
        "Communication module not working, what should I check?",
    ]
    for q in test_queries:
        print(f"\n{'=' * 60}\nQuery: {q}\n{'=' * 60}")
        crew = build_crew(q)
        result = crew.kickoff()
        print(f"\nFinal result:\n{result}")