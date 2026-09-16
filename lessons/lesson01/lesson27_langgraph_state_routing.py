# ============================================================
# LESSON 27 — LANGGRAPH STATE + CONDITIONAL ROUTING
#
# Goal:
# Combine two LangGraph concepts:
#
# 1. STATE
#    Carries information through the workflow.
#
# 2. CONDITIONAL EDGE
#    Decides which path the workflow should follow.
#
#
# RUNTIME ARCHITECTURE:
#
#                         ┌──→ TOOL NODE ──→ FINAL
#                         │
# START → DECISION NODE ──┤
#                         │
#                         └──→ DIRECT NODE ─→ FINAL
#
#
# The Decision Node updates State.
# The Conditional Edge reads State and chooses the path.
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
# State is the information carried through the graph.
#
# question  → user's question
# route     → decision about which path to take
# result    → result produced by the selected path
# answer    → final response
# ============================================================

class AgentState(TypedDict):

    question: str
    route: str
    result: str
    answer: str


# ============================================================
# SETUP / DEFINITION
# 3. DECISION NODE
#
# This Node examines the question and updates State.
#
# In this lesson we use a simple Python rule.
#
# Later, an LLM can make this decision.
# ============================================================

def decision_node(state: AgentState):

    print("\nDECISION NODE")

    question = state["question"].lower()

    if "employee" in question or "salary" in question:

        print("Decision: TOOL path")

        return {
            "route": "tool"
        }

    else:

        print("Decision: DIRECT path")

        return {
            "route": "direct"
        }


# ============================================================
# SETUP / DEFINITION
# 4. ROUTING FUNCTION
#
# This function is used by the Conditional Edge.
#
# It reads the State and tells LangGraph which path to follow.
#
# IMPORTANT:
# This is a routing function, NOT a Node.
# ============================================================

def route_question(state: AgentState):

    if state["route"] == "tool":

        return "tool"

    return "direct"


# ============================================================
# SETUP / DEFINITION
# 5. TOOL NODE
#
# This represents a path where a Tool would be used.
#
# For now, we use a simple Python operation.
# ============================================================

def tool_node(state: AgentState):

    print("TOOL NODE: Executing tool")

    return {
        "result": (
            "Tool processed the question: "
            + state["question"]
        )
    }


# ============================================================
# SETUP / DEFINITION
# 6. DIRECT NODE
#
# This represents a path where no Tool is required.
# ============================================================

def direct_node(state: AgentState):

    print("DIRECT NODE: No tool required")

    return {
        "result": (
            "Question can be answered directly: "
            + state["question"]
        )
    }


# ============================================================
# SETUP / DEFINITION
# 7. FINAL NODE
#
# This Node reads the result from State and prepares
# the final response.
# ============================================================

def final_node(state: AgentState):

    print("FINAL NODE: Preparing final answer")

    return {
        "answer": state["result"]
    }


# ============================================================
# SETUP / DEFINITION
# 8. CREATE THE GRAPH
# ============================================================

graph = StateGraph(AgentState)


# ============================================================
# SETUP / DEFINITION
# 9. ADD NODES
#
# Nodes:
#
# decision_node
# tool_node
# direct_node
# final_node
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
    "direct_node",
    direct_node
)

graph.add_node(
    "final_node",
    final_node
)


# ============================================================
# SETUP / DEFINITION
# 10. START EDGE
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
# 11. CONDITIONAL EDGE
#
# Decision Node
#       ↓
# route_question()
#       ↓
# ┌─────┴─────┐
# ↓           ↓
# Tool       Direct
#
# The value returned by route_question() determines
# which Node executes next.
# ============================================================

graph.add_conditional_edges(
    "decision_node",
    route_question,
    {
        "tool": "tool_node",
        "direct": "direct_node"
    }
)


# ============================================================
# SETUP / DEFINITION
# 12. CONNECT BOTH PATHS TO FINAL NODE
#
# Tool Path:
#
# Tool Node
#     ↓
# Final Node
#
#
# Direct Path:
#
# Direct Node
#     ↓
# Final Node
# ============================================================

graph.add_edge(
    "tool_node",
    "final_node"
)

graph.add_edge(
    "direct_node",
    "final_node"
)


# ============================================================
# SETUP / DEFINITION
# 13. FINAL NODE → END
# ============================================================

graph.add_edge(
    "final_node",
    END
)


# ============================================================
# SETUP / DEFINITION
# 14. COMPILE GRAPH
# ============================================================

app = graph.compile()


# ============================================================
# RUNTIME FLOW
# 15. GET USER QUESTION
# ============================================================

question = input(
    "\nAsk a question: "
)


# ============================================================
# RUNTIME FLOW
# 16. INITIAL STATE
#
# The question enters the graph.
# Other fields start empty.
# ============================================================

initial_state = {
    "question": question,
    "route": "",
    "result": "",
    "answer": ""
}


# ============================================================
# RUNTIME FLOW
# 17. EXECUTE GRAPH
#
# The workflow is:
#
# START
#   ↓
# Decision Node
#   ↓
# Conditional Edge
#   ↓
# ┌───────────────┐
# │               │
# Tool          Direct
# │               │
# └───────┬───────┘
#         ↓
#     Final Node
#         ↓
#        END
# ============================================================

result = app.invoke(
    initial_state
)


# ============================================================
# RUNTIME FLOW
# 18. FINAL STATE
# ============================================================

print("\nFINAL STATE:")

print(result)


# ============================================================
# RUNTIME FLOW
# 19. FINAL ANSWER
# ============================================================

print("\nFINAL ANSWER:")

print(result["answer"])