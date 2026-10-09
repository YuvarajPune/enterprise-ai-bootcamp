"""
Lesson 48.2 — Basic Tracing

Goal:
Understand how tracing records the journey of an agent workflow.

Mental model:

    LLM
     ↓
    LangGraph
     ↓
    Tool
     ↓
    Result
     ↓
    Final Answer

Tracing helps us see this journey step by step.
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# ============================================================
# STEP 1 — DEFINE WORKFLOW STATE
# ============================================================

class AgentState(TypedDict):
    user_request: str
    decision: str
    tool_result: str
    final_answer: str


# ============================================================
# STEP 2 — LLM REASONING
# ============================================================

def llm_reason(state: AgentState):

    print("\n[TRACE] LLM REASONING")
    print(f"[TRACE] User request: {state['user_request']}")

    # In this lesson we simulate the LLM decision.
    # Later we will connect a real LLM.
    decision = "check_product"

    print(f"[TRACE] LLM decision: {decision}")

    return {
        "decision": decision
    }


# ============================================================
# STEP 3 — TOOL EXECUTION
# ============================================================

def execute_tool(state: AgentState):

    print("\n[TRACE] TOOL EXECUTION")

    print("[TRACE] Tool: Product MCP")
    print("[TRACE] Input: MacBook Air")

    # Simulated tool result
    result = "MacBook Air is available."

    print(f"[TRACE] Tool result: {result}")

    return {
        "tool_result": result
    }


# ============================================================
# STEP 4 — FINAL ANSWER
# ============================================================

def final_answer(state: AgentState):

    print("\n[TRACE] FINAL ANSWER")

    answer = state["tool_result"]

    print(f"[TRACE] Final answer: {answer}")

    return {
        "final_answer": answer
    }


# ============================================================
# STEP 5 — BUILD LANGGRAPH
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
# STEP 6 — RUN WORKFLOW
# ============================================================

print("=" * 60)
print("STARTING AGENT")
print("=" * 60)

result = app.invoke({
    "user_request": "Is MacBook Air available?"
})

print("\n" + "=" * 60)
print("FINAL STATE")
print("=" * 60)

print(result)