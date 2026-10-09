"""
Lesson 48.8 — Tracing a Real Tool Failure

Goal:
Capture SUCCESS and FAILED operations inside a trace.

Mental model:

    LLM
     ↓
    LangGraph
     ↓
    MCP / Tool
     ↓
    ❌ Failure
     ↓
    Trace records the failure

Important:
Tracing records the failure.
Recovery logic decides what to do about it.
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
    tool_error: str
    final_answer: str


# ============================================================
# STEP 2 — START A SPAN
# ============================================================

def start_span(trace_id, span_name):

    span_id = str(uuid.uuid4())

    print(f"\n[TRACE {trace_id}] START SPAN")
    print(f"  Span ID: {span_id}")
    print(f"  Span: {span_name}")

    return span_id


# ============================================================
# STEP 3 — END A SUCCESSFUL SPAN
# ============================================================

def end_span_success(trace_id, span_id, span_name, start_time):

    duration = time.time() - start_time

    print(f"[TRACE {trace_id}] END SPAN")
    print(f"  Span ID: {span_id}")
    print(f"  Span: {span_name}")
    print("  Status: SUCCESS")
    print(f"  Duration: {duration:.3f} seconds")


# ============================================================
# STEP 4 — END A FAILED SPAN
# ============================================================

def end_span_failure(
    trace_id,
    span_id,
    span_name,
    start_time,
    error
):

    duration = time.time() - start_time

    print(f"[TRACE {trace_id}] END SPAN")
    print(f"  Span ID: {span_id}")
    print(f"  Span: {span_name}")
    print("  Status: FAILED")
    print(f"  Error: {error}")
    print(f"  Duration: {duration:.3f} seconds")


# ============================================================
# STEP 5 — LLM REASONING
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

    end_span_success(
        trace_id,
        span_id,
        "LLM Reasoning",
        start_time
    )

    return {
        "decision": decision
    }


# ============================================================
# STEP 6 — LANGGRAPH ROUTING
# ============================================================

def route_request(state: AgentState):

    trace_id = state["trace_id"]
    start_time = time.time()

    span_id = start_span(
        trace_id,
        "LangGraph Routing"
    )

    print(f"  Routing decision: {state['decision']}")

    end_span_success(
        trace_id,
        span_id,
        "LangGraph Routing",
        start_time
    )

    return {}


# ============================================================
# STEP 7 — PRODUCT MCP
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

    try:

        # ----------------------------------------------------
        # Simulate an MCP failure
        # ----------------------------------------------------

        raise RuntimeError(
            "Product service unavailable"
        )

    except Exception as error:

        print(f"  MCP error: {error}")

        end_span_failure(
            trace_id,
            span_id,
            "Product MCP",
            start_time,
            str(error)
        )

        return {
            "tool_result": "",
            "tool_error": str(error)
        }


# ============================================================
# STEP 8 — FINAL ANSWER
# ============================================================

def final_answer(state: AgentState):

    trace_id = state["trace_id"]
    start_time = time.time()

    span_id = start_span(
        trace_id,
        "Final Answer"
    )

    if state["tool_error"]:

        answer = (
            "The MacBook Air availability check "
            "failed because the product service "
            "was unavailable."
        )

    else:

        answer = state["tool_result"]

    print(f"  Answer: {answer}")

    end_span_success(
        trace_id,
        span_id,
        "Final Answer",
        start_time
    )

    return {
        "final_answer": answer
    }


# ============================================================
# STEP 9 — BUILD LANGGRAPH
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
# STEP 10 — CREATE TRACE ID
# ============================================================

trace_id = "TR-" + str(uuid.uuid4())[:8]

print("=" * 60)
print(f"START TRACE: {trace_id}")
print("=" * 60)


# ============================================================
# STEP 11 — RUN WORKFLOW
# ============================================================

result = app.invoke({
    "trace_id": trace_id,
    "user_request": "Is MacBook Air available?",
    "tool_result": "",
    "tool_error": "",
    "final_answer": ""
})


# ============================================================
# STEP 12 — FINAL STATE
# ============================================================

print("\n" + "=" * 60)
print("FINAL STATE")
print("=" * 60)

print(result)
