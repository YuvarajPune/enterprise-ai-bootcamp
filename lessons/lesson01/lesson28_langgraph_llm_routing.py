# ============================================================
# LESSON 28 — LANGGRAPH LLM-BASED ROUTING
#
# Goal:
# Let the LLM decide which path the LangGraph workflow should
# take.
#
# Previous lesson:
#   Python logic decided the route.
#
# This lesson:
#   LLM decides the route.
#
#
# RUNTIME ARCHITECTURE:
#
#                     ┌──→ TOOL NODE ──→ FINAL LLM
#                     │
# START → LLM DECISION
#                     │
#                     └──→ DIRECT ANSWER
#                              ↓
#                            END
#
# Key concepts:
#   LLM          = makes the decision
#   State        = carries the decision
#   Conditional Edge = routes based on the decision
#   Node         = performs the actual work
# ============================================================


# ============================================================
# SETUP / DEFINITION
# 1. IMPORTS
# ============================================================

from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq


# ============================================================
# SETUP / DEFINITION
# 2. LLM — REASONING ENGINE
#
# The LLM will decide whether the question requires a Tool.
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=None
)


# ============================================================
# SETUP / DEFINITION
# 3. STATE
#
# State carries information through the graph.
#
# question → user's question
# route    → decision made by the LLM
# result   → result produced by the selected path
# answer   → final answer
# ============================================================

class AgentState(TypedDict):

    question: str
    route: str
    result: str
    answer: str


# ============================================================
# SETUP / DEFINITION
# 4. LLM DECISION NODE
#
# The LLM receives the user's question and decides:
#
#   "tool"   → a Tool is required
#   "direct" → no Tool is required
#
# IMPORTANT:
# This Node uses the LLM for the decision.
# ============================================================

def llm_decision_node(state: AgentState):

    print("\nLLM DECISION NODE")

    question = state["question"]

    prompt = f"""
You are deciding how to handle a user question.

User question:
{question}

If the question requires employee information or
salary information, respond with exactly:

tool

Otherwise respond with exactly:

direct

Do not provide any other text.
"""

    response = llm.invoke(prompt)

    route = response.content.strip().lower()

    # Safety check in case the LLM returns unexpected text.
    if route not in ["tool", "direct"]:
        route = "direct"

    print("LLM decision:", route)

    return {
        "route": route
    }


# ============================================================
# SETUP / DEFINITION
# 5. CONDITIONAL ROUTING FUNCTION
#
# This function reads the route stored in State.
#
# IMPORTANT:
# The LLM made the decision.
#
# This function only tells LangGraph where to go.
# ============================================================

def route_question(state: AgentState):

    if state["route"] == "tool":

        return "tool"

    return "direct"


# ============================================================
# SETUP / DEFINITION
# 6. TOOL NODE
#
# This represents the Tool path.
#
# In a real Agent this could be:
#
#   SQL Tool
#   MCP Tool
#   API Tool
#   RAG Tool
# ============================================================

def tool_node(state: AgentState):

    print("TOOL NODE: Executing Tool")

    return {
        "result": (
            "Tool processed the question: "
            + state["question"]
        )
    }


# ============================================================
# SETUP / DEFINITION
# 7. DIRECT NODE
#
# This path is used when the LLM decides that no Tool
# is required.
# ============================================================

def direct_node(state: AgentState):

    print("DIRECT NODE: No Tool required")

    return {
        "result": (
            "The question can be answered directly."
        )
    }


# ============================================================
# SETUP / DEFINITION
# 8. FINAL LLM NODE
#
# The LLM receives the result from the selected path
# and prepares the final answer.
# ============================================================

def final_llm_node(state: AgentState):

    print("FINAL LLM NODE: Preparing final answer")

    prompt = f"""
Answer the user's question.

User question:
{state["question"]}

Information available:
{state["result"]}

Give a short, clear answer.
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


# ============================================================
# SETUP / DEFINITION
# 9. CREATE THE GRAPH
# ============================================================

graph = StateGraph(AgentState)


# ============================================================
# SETUP / DEFINITION
# 10. ADD NODES
#
# Nodes:
#
# 1. LLM Decision Node
# 2. Tool Node
# 3. Direct Node
# 4. Final LLM Node
# ============================================================

graph.add_node(
    "llm_decision_node",
    llm_decision_node
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
    "final_llm_node",
    final_llm_node
)


# ============================================================
# SETUP / DEFINITION
# 11. START → LLM DECISION
#
# The workflow starts with the LLM decision.
# ============================================================

graph.add_edge(
    START,
    "llm_decision_node"
)


# ============================================================
# SETUP / DEFINITION
# 12. CONDITIONAL EDGE
#
# LLM Decision
#       ↓
#   route_question()
#       ↓
# ┌─────┴─────┐
# ↓           ↓
# Tool       Direct
#
# The route stored in State determines the path.
# ============================================================

graph.add_conditional_edges(
    "llm_decision_node",
    route_question,
    {
        "tool": "tool_node",
        "direct": "direct_node"
    }
)


# ============================================================
# SETUP / DEFINITION
# 13. TOOL / DIRECT → FINAL LLM
#
# Both paths eventually come back to the Final LLM Node.
# ============================================================

graph.add_edge(
    "tool_node",
    "final_llm_node"
)

graph.add_edge(
    "direct_node",
    "final_llm_node"
)


# ============================================================
# SETUP / DEFINITION
# 14. FINAL LLM → END
# ============================================================

graph.add_edge(
    "final_llm_node",
    END
)


# ============================================================
# SETUP / DEFINITION
# 15. COMPILE GRAPH
# ============================================================

app = graph.compile()


# ============================================================
# RUNTIME FLOW
# 16. GET USER QUESTION
# ============================================================

question = input(
    "\nAsk a question: "
)


# ============================================================
# RUNTIME FLOW
# 17. INITIAL STATE
# ============================================================

initial_state = {
    "question": question,
    "route": "",
    "result": "",
    "answer": ""
}


# ============================================================
# RUNTIME FLOW
# 18. EXECUTE GRAPH
#
#                     ┌──→ Tool → Final LLM
#                     │
# START → LLM Decision
#                     │
#                     └──→ Direct → Final LLM
#                                      ↓
#                                     END
# ============================================================

result = app.invoke(
    initial_state
)


# ============================================================
# RUNTIME FLOW
# 19. FINAL STATE
# ============================================================

print("\nFINAL STATE:")

print(result)


# ============================================================
# RUNTIME FLOW
# 20. FINAL ANSWER
# ============================================================

print("\nFINAL ANSWER:")

print(result["answer"])