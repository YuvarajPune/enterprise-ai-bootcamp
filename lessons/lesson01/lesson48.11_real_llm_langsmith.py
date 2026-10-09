"""
Lesson 48.11 — Real LLM Tracing in LangSmith

Goal:
Use a real LLM inside LangGraph and observe
the LLM call through LangSmith.

Mental model:

    User Request
         ↓
       LLM
    (Reasoning)
         ↓
    LangGraph
         ↓
       Tool

LangSmith observes the execution.
"""

from typing import TypedDict
import os
import time

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from google import genai


# ============================================================
# STEP 1 — LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

# .env contains:
#
# GOOGLE_API_KEY=your Gemini API key
# LANGSMITH_TRACING=true
# LANGSMITH_API_KEY=your LangSmith key
# LANGSMITH_PROJECT=agentic-ai-learning


# ============================================================
# STEP 2 — CREATE GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)


# ============================================================
# STEP 3 — WORKFLOW STATE
# ============================================================

class AgentState(TypedDict, total=False):

    user_request: str
    decision: str
    tool_result: str
    final_answer: str


# ============================================================
# STEP 4 — REAL LLM REASONING
# ============================================================

def llm_reason(state: AgentState):

    print("\nLLM: Thinking...")

    prompt = f"""
You are the reasoning component of an AI agent.

User request:
{state["user_request"]}

Decide what the agent should do next.

If the request is asking about product availability,
respond with exactly:

check_product

Otherwise respond with:

unknown
"""

    start_time = time.time()

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    duration = time.time() - start_time

    decision = response.text.strip()

    print(f"LLM decision: {decision}")
    print(f"LLM time: {duration:.3f} seconds")

    return {
        "decision": decision
    }


# ============================================================
# STEP 5 — TOOL EXECUTION
# ============================================================

def execute_tool(state: AgentState):

    print("\nTOOL: Checking MacBook Air...")

    result = "MacBook Air is available."

    return {
        "tool_result": result
    }


# ============================================================
# STEP 6 — FINAL ANSWER
# ============================================================

def final_answer(state: AgentState):

    answer = state["tool_result"]

    print("\nFINAL ANSWER:")
    print(answer)

    return {
        "final_answer": answer
    }


# ============================================================
# STEP 7 — BUILD LANGGRAPH
# ============================================================

graph = StateGraph(AgentState)

graph.add_node("llm_reason", llm_reason)
graph.add_node("execute_tool", execute_tool)
graph.add_node("final_answer", final_answer)

graph.add_edge(START, "llm_reason")
graph.add_edge("llm_reason", "execute_tool")
graph.add_edge("execute_tool", "final_answer")
graph.add_edge("final_answer", END)

app = graph.compile()


# ============================================================
# STEP 8 — RUN WORKFLOW
# ============================================================

print("=" * 60)
print("STARTING REAL LLM WORKFLOW")
print("=" * 60)

result = app.invoke({
    "user_request": "Is MacBook Air available?"
})


# ============================================================
# STEP 9 — FINAL STATE
# ============================================================

print("\n" + "=" * 60)
print("FINAL STATE")
print("=" * 60)

print(result)

print("\nWaiting for LangSmith tracing to finish...")

time.sleep(5)

print("Done.")