"""
Lesson 48.6 — Parent Trace + Child Spans

Goal:
Understand how a trace contains multiple spans.

Mental model:

    TRACE
       |
       +--- LLM span
       |
       +--- LangGraph span
       |
       +--- Tool span
       |
       +--- Final Answer span
"""

from typing import TypedDict
import uuid
import time

from langgraph.graph import StateGraph, START, END


# ============================================================
# STEP 1 — WORKFLOW STATE
# ============================================================

class AgentState(TypedDict):
    trace_id: str
    user_request: str
    decision: str
    tool_result: str
    final_answer: str


# ============================================================
# STEP 2 — SIMPLE SPAN FUNCTION
# ============================================================

def start_span(trace_id, span_name):
    """
    Start one operation inside the trace.

    trace_id  = identifies the complete workflow
    span_name = identifies the current operation
    """

    span_id = str(uuid.uuid4())

    print(f"\n[TRACE {trace_id}] START SPAN")
    print(f"  Span ID: {span_id}")
    print(f"  Span: {span_name}")

    return span_id


def end_span(trace_id, span_id, span_name, start_time):
    """
    Finish the span and calculate how long it took.
    """

    duration = time.time() - start_time

    print(f"[TRACE {trace_id}] END SPAN")
    print(f"  Span ID: {span_id}")
    print(f"  Span: {span_name}")
    print(f"  Duration: {duration:.3f} seconds")


# ============================================================
# STEP 3 — LLM NODE
# ============================================================

def llm_reason(state: AgentState):

    trace_id = state["trace_id"]

    start_time = time.time()

    span_id = start_span(
        trace_id,
        "LLM Reasoning"
    )

    # Simulated LLM decision
    decision = "check_product"

    print(f"  LLM decision: {decision}")

    end_span(
        trace_id,
        span_id,
        "LLM Reasoning",
        start_time
    )

    return {
        "decision": decision
    }


# ============================================================
# STEP 4 — LANGGRAPH NODE
# ============================================================

def route_request(state: AgentState):

    trace_id = state["trace_id"]

    start_time = time.time()

    span_id = start_span(
        trace_id,
        "LangGraph Routing"
    )

    decision = state["decision"]

    print(f"  Routing decision: {decision}")

    end_span(
        trace_id,
        span_id,
        "LangGraph Routing",
        start_time
    )

    return {}


# ============================================================
# STEP 5 — TOOL NODE
# ============================================================

def execute_tool(state: AgentState):

    trace_id = state["trace_id"]

    start_time = time.time()

    span_id = start_span(
        trace_id,
        "Product MCP"
    )

    print("  Tool: Product MCP")
    print("  Input: MacBook Air")

    # Simulated MCP result
    result = "MacBook Air is available."

    print(f"  Tool result: {result}")

    end_span(
        trace_id,
        span_id,
        "Product MCP",
        start_time
    )

    return {
        "tool_result": result
    }


# ============================================================
# STEP 6 — FINAL ANSWER NODE
# ============================================================

def final_answer(state: AgentState):

    trace_id = state["trace_id"]

    start_time = time.time()

    span_id = start_span(
        trace_id,
        "Final Answer"
    )

    answer = state["tool_result"]

    print(f"  Answer: {answer}")

    end_span(
        trace_id,
        span_id,
        "Final Answer",
        start_time
    )

    return {
        "final_answer": answer
    }


# ============================================================
# STEP 7 — BUILD LANGGRAPH
# ============================================================

graph = StateGraph(AgentState)

graph.add_node("llm_reason", llm_reason)
graph.add_node("route_request", route_request)
graph.add_node("execute_tool", execute_tool)
graph.add_node("final_answer", final_answer)

graph.add_edge(START, "llm_reason")
graph.add_edge("llm_reason", "route_request")
graph.add_edge("route_request", "execute_tool")
graph.add_edge("execute_tool", "final_answer")
graph.add_edge("final_answer", END)

app = graph.compile()


# ============================================================
# STEP 8 — CREATE TRACE ID
# ============================================================

trace_id = "TR-" + str(uuid.uuid4())[:8]

print("=" * 60)
print(f"START TRACE: {trace_id}")
print("=" * 60)


# ============================================================
# STEP 9 — RUN WORKFLOW
# ============================================================

result = app.invoke({
    "trace_id": trace_id,
    "user_request": "Is MacBook Air available?"
})


# ============================================================
# STEP 10 — FINAL STATE
# ============================================================

print("\n" + "=" * 60)
print("FINAL STATE")
print("=" * 60)

print(result)