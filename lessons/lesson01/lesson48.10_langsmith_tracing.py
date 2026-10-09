"""
Lesson 48.10 — LangSmith + LangGraph

Goal:
Connect LangGraph to LangSmith and observe the workflow.

LangSmith handles tracing automatically.

Mental model:

    LLM
     ↓
    LangGraph
     ↓
    Tool
     ↓
    Result

            ↓
       LangSmith
            ↓
       Observe
"""

from typing import TypedDict
import time

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END


# ============================================================
# STEP 1 — LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

# LangSmith reads:
#
# LANGSMITH_TRACING
# LANGSMITH_API_KEY
# LANGSMITH_PROJECT
#
# from the environment.


# ============================================================
# STEP 2 — WORKFLOW STATE
# ============================================================

class AgentState(TypedDict, total=False):

    user_request: str
    decision: str
    tool_result: str


# ============================================================
# STEP 3 — LLM REASONING NODE
# ============================================================

def llm_reason(state: AgentState):

    print("\nLLM: Reasoning about the request...")

    # Simulated LLM decision
    decision = "check_product"

    return {
        "decision": decision
    }


# ============================================================
# STEP 4 — TOOL EXECUTION NODE
# ============================================================

def execute_tool(state: AgentState):

    print("\nTOOL: Checking MacBook Air...")

    result = "MacBook Air is available."

    return {
        "tool_result": result
    }


# ============================================================
# STEP 5 — BUILD LANGGRAPH
# ============================================================

graph = StateGraph(AgentState)

graph.add_node("llm_reason", llm_reason)
graph.add_node("execute_tool", execute_tool)

graph.add_edge(START, "llm_reason")
graph.add_edge("llm_reason", "execute_tool")
graph.add_edge("execute_tool", END)

app = graph.compile()


# ============================================================
# STEP 6 — RUN WORKFLOW
# ============================================================

print("=" * 60)
print("STARTING LANGGRAPH WORKFLOW")
print("=" * 60)

result = app.invoke({
    "user_request": "Is MacBook Air available?"
})


# ============================================================
# STEP 7 — FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("FINAL RESULT")
print("=" * 60)

print(result)


# ============================================================
# STEP 8 — GIVE LANGSMITH TIME TO SEND TRACE
# ============================================================

print("\nWaiting for LangSmith tracing to finish...")

time.sleep(5)

print("Done.")