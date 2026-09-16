# ============================================================
# LESSON 25 — LANGGRAPH CONDITIONAL EDGES
#
# Goal:
# Understand how LangGraph can choose different paths
# based on the current State.
#
# Key concepts:
#   Node              = performs a task
#   Edge              = fixed path
#   Conditional Edge  = chooses the next path
#   State             = information used for the decision
#
# RUNTIME ARCHITECTURE:
#
#                 ┌──→ TOOL NODE ──→ FINAL NODE
#                 │
# START → DECISION NODE
#                 │
#                 └──→ FINAL NODE
#
# This lesson uses a simple Python decision function.
# Later, the LLM itself will make the decision.
# ============================================================


# ============================================================
# SETUP / DEFINITION
# 1. IMPORTS
# ============================================================

from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# ============================================================
# SETUP / DEFINITION
# 2. STATE
#
# The State carries the question and the decision.
# ============================================================

class AgentState(TypedDict):

    question: str
    needs_tool: bool
    tool_result: str
    answer: str


# ============================================================
# SETUP / DEFINITION
# 3. DECISION NODE
#
# This Node examines the question and determines whether
# a Tool is required.
#
# For now, this is a simple Python rule.
# ============================================================

def decision_node(state: AgentState):

    question = state["question"].lower()

    if "calculate" in question or "employee" in question:

        return {
            "needs_tool": True
        }

    return {
        "needs_tool": False
    }


# ============================================================
# SETUP / DEFINITION
# 4. CONDITIONAL ROUTING FUNCTION
#
# This function looks at State and tells LangGraph
# which path to follow next.
#
# IMPORTANT:
#
# This is NOT a Node.
#
# It is a routing function used by a Conditional Edge.
# ============================================================

def route_question(state: AgentState):

    if state["needs_tool"]:
        return "tool"

    return "direct"


# ============================================================
# SETUP / DEFINITION
# 5. TOOL NODE
#
# This Node represents a Tool execution.
# ============================================================

def tool_node(state: AgentState):

    print("\nTOOL NODE: Tool is being used")

    return {
        "tool_result": (
            "Tool was used to process: "
            + state["question"]
        )
    }


# ============================================================
# SETUP / DEFINITION
# 6. FINAL NODE
#
# This Node prepares the final response.
# ============================================================

def final_node(state: AgentState):

    print("\nFINAL NODE: Preparing answer")

    if state["tool_result"]:

        answer = (
            "I used a tool to process your question.\n"
            + state["tool_result"]
        )

    else:

        answer = (
            "I can answer this question directly."
        )

    return {
        "answer": answer
    }


# ============================================================
# SETUP / DEFINITION
# 7. CREATE THE GRAPH
# ============================================================

graph = StateGraph(AgentState)


# ============================================================
# SETUP / DEFINITION
# 8. ADD NODES
# ============================================================

graph.add_node(
    "decision_node",
    decision_node
)

graph.add_node(
    "tool_node",
    tool_node
)

graph.add_node(
    "final_node",
    final_node
)


# ============================================================
# SETUP / DEFINITION
# 9. ADD START EDGE
#
# START
#   ↓
# Decision Node
# ============================================================

graph.add_edge(
    START,
    "decision_node"
)


# ============================================================
# SETUP / DEFINITION
# 10. ADD CONDITIONAL EDGE
#
# Decision Node
#       ↓
#   route_question()
#       ↓
# ┌─────┴─────┐
# ↓           ↓
# Tool       Final
#
# The value returned by route_question() determines
# which Node executes next.
# ============================================================

graph.add_conditional_edges(
    "decision_node",
    route_question,
    {
        "tool": "tool_node",
        "direct": "final_node"
    }
)


# ============================================================
# SETUP / DEFINITION
# 11. CONNECT TOOL NODE TO FINAL NODE
#
# Tool
#   ↓
# Final
# ============================================================

graph.add_edge(
    "tool_node",
    "final_node"
)


# ============================================================
# SETUP / DEFINITION
# 12. CONNECT FINAL NODE TO END
# ============================================================

graph.add_edge(
    "final_node",
    END
)


# ============================================================
# SETUP / DEFINITION
# 13. COMPILE GRAPH
# ============================================================

app = graph.compile()


# ============================================================
# RUNTIME FLOW
# 14. GET USER QUESTION
# ============================================================

question = input(
    "\nAsk a question: "
)


# ============================================================
# RUNTIME FLOW
# 15. INITIAL STATE
# ============================================================

initial_state = {
    "question": question,
    "needs_tool": False,
    "tool_result": "",
    "answer": ""
}


# ============================================================
# RUNTIME FLOW
# 16. EXECUTE GRAPH
#
# The graph will choose one of two paths:
#
# PATH 1:
# START
#   ↓
# Decision
#   ↓
# Tool
#   ↓
# Final
#   ↓
# END
#
# PATH 2:
# START
#   ↓
# Decision
#   ↓
# Final
#   ↓
# END
# ============================================================

result = app.invoke(
    initial_state
)


# ============================================================
# RUNTIME FLOW
# 17. FINAL STATE
# ============================================================

print("\nFINAL STATE:")

print(result)


# ============================================================
# RUNTIME FLOW
# 18. FINAL ANSWER
# ============================================================

print("\nFINAL ANSWER:")

print(result["answer"])