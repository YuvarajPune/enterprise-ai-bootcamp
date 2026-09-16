# ============================================================
# LESSON 29 — LANGGRAPH AGENT LOOP
#
# Goal:
# Understand how an Agent can:
#
#   1. Ask the LLM what to do
#   2. Execute a Tool
#   3. Return the Tool result to the LLM
#   4. Decide whether the task is complete
#   5. Stop when the task is complete
#
#
# RUNTIME ARCHITECTURE:
#
#                  ┌──────────────────────┐
#                  │                      │
#                  ↓                      │
# START → LLM → DECISION                 │
#            │       │                    │
#            │       │ TOOL               │
#            │       ↓                    │
#            │     TOOL ──────────────────┘
#            │
#            │ FINAL
#            ↓
#          FINAL
#            ↓
#           END
#
#
# IMPORTANT:
# The loop MUST have a termination condition.
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
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=None
)


# ============================================================
# SETUP / DEFINITION
# 3. STATE
#
# State carries information through the Agent loop.
#
# question    → user's question
# tool_count  → number of Tool calls
# tool_result → latest Tool result
# answer      → final answer
# ============================================================

class AgentState(TypedDict):

    question: str
    tool_count: int
    tool_result: str
    answer: str


# ============================================================
# SETUP / DEFINITION
# 4. LLM NODE
#
# The LLM examines the current State and decides:
#
#   "tool"  → use the Tool
#   "final" → task is complete
#
# IMPORTANT:
# After the Tool executes, this Node runs again.
# ============================================================

def llm_node(state: AgentState):

    print("\nLLM NODE: Thinking")

    # --------------------------------------------------------
    # TERMINATION CONDITION
    #
    # Once the Tool has been executed, we consider the
    # demonstration task complete.
    #
    # This prevents an uncontrolled loop.
    # --------------------------------------------------------

    if state["tool_count"] >= 1:

        decision = "final"

    else:

        prompt = f"""
You are an Agent deciding how to handle a user question.

User question:
{state["question"]}

No Tool has been used yet.

If the question requires employee information,
respond with exactly:

tool

Otherwise respond with exactly:

final

Do not provide any other text.
"""

        response = llm.invoke(prompt)

        decision = response.content.strip().lower()

        if decision not in ["tool", "final"]:
            decision = "final"

    print("LLM decision:", decision)

    return {
        "tool_result": decision
    }


# ============================================================
# SETUP / DEFINITION
# 5. CONDITIONAL ROUTING FUNCTION
#
# The routing function reads the LLM's decision from State.
#
# It tells LangGraph where to go next.
# ============================================================

def route_after_llm(state: AgentState):

    if state["tool_result"] == "tool":

        return "tool"

    return "final"


# ============================================================
# SETUP / DEFINITION
# 6. TOOL NODE
#
# This represents execution of a Tool.
#
# In a real enterprise Agent this could be:
#
#   SQL Tool
#   MCP Tool
#   API Tool
#   RAG Tool
# ============================================================

def tool_node(state: AgentState):

    print("\nTOOL NODE: Executing Tool")

    new_count = state["tool_count"] + 1

    print("Tool call number:", new_count)

    return {
        "tool_count": new_count,
        "tool_result": (
            "Employee information retrieved successfully."
        )
    }


# ============================================================
# SETUP / DEFINITION
# 7. FINAL NODE
#
# This Node prepares the final response.
# ============================================================

def final_node(state: AgentState):

    print("\nFINAL NODE: Preparing answer")

    if state["tool_count"] > 0:

        answer = (
            "The Agent used the Tool and completed the task."
        )

    else:

        answer = (
            "The Agent answered without using a Tool."
        )

    return {
        "answer": answer
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
#   llm_node
#   tool_node
#   final_node
# ============================================================

graph.add_node(
    "llm_node",
    llm_node
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
# 10. START → LLM
# ============================================================

graph.add_edge(
    START,
    "llm_node"
)


# ============================================================
# SETUP / DEFINITION
# 11. LLM → CONDITIONAL ROUTING
#
# The LLM decides:
#
#   tool  → Tool Node
#   final → Final Node
# ============================================================

graph.add_conditional_edges(
    "llm_node",
    route_after_llm,
    {
        "tool": "tool_node",
        "final": "final_node"
    }
)


# ============================================================
# SETUP / DEFINITION
# 12. TOOL → LLM
#
# THIS CREATES THE AGENT LOOP.
#
# Tool result is placed into State.
# Then execution returns to the LLM.
#
# Tool
#   ↓
# LLM
# ============================================================

graph.add_edge(
    "tool_node",
    "llm_node"
)


# ============================================================
# SETUP / DEFINITION
# 13. FINAL → END
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
# ============================================================

initial_state = {
    "question": question,
    "tool_count": 0,
    "tool_result": "",
    "answer": ""
}


# ============================================================
# RUNTIME FLOW
# 17. EXECUTE AGENT LOOP
#
# Example for an employee question:
#
# START
#   ↓
# LLM
#   ↓
# "tool"
#   ↓
# TOOL
#   ↓
# State updated
#   ↓
# LLM
#   ↓
# "final"
#   ↓
# FINAL
#   ↓
# END
#
# The second LLM decision is what stops the loop.
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