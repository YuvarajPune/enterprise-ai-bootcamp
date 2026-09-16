# ============================================================
# LESSON 24 — LANGGRAPH BASICS
#
# Goal:
# Connect an LLM Node and a Tool Node using LangGraph.
#
# RUNTIME ARCHITECTURE:
#
# START
#   ↓
# LLM Node
#   ↓
# Tool Node
#   ↓
# LLM Node
#   ↓
# END
#
# Key concepts:
#   Node = performs a task
#   Edge = controls the path
#   State = carries information
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
# The State carries information between Nodes.
#
# question → user's question
# answer   → final answer
# tool_result → result produced by the Tool
# ============================================================

class AgentState(TypedDict):

    question: str
    tool_result: str
    answer: str


# ============================================================
# SETUP / DEFINITION
# 4. LLM NODE — FIRST STEP
#
# The LLM receives the user's question.
#
# In this simple demonstration, the LLM prepares the
# information that will be passed to the Tool.
# ============================================================

def llm_node(state: AgentState):

    print("\nLLM NODE: Processing question")

    return {
        "tool_result": f"Information requested for: {state['question']}"
    }


# ============================================================
# SETUP / DEFINITION
# 5. TOOL NODE
#
# A Node can execute a Python function/tool.
#
# This is a very simple Tool for demonstration.
# ============================================================

def tool_node(state: AgentState):

    print("TOOL NODE: Executing tool")

    result = f"Tool processed → {state['tool_result']}"

    return {
        "tool_result": result
    }


# ============================================================
# SETUP / DEFINITION
# 6. LLM NODE — FINAL STEP
#
# The LLM receives the Tool result and prepares the answer.
# ============================================================

def final_llm_node(state: AgentState):

    print("LLM NODE: Preparing final answer")

    response = llm.invoke(
        f"Give a short answer based on this tool result: "
        f"{state['tool_result']}"
    )

    return {
        "answer": response.content
    }


# ============================================================
# SETUP / DEFINITION
# 7. CREATE THE GRAPH
# ============================================================

graph = StateGraph(AgentState)


# ============================================================
# SETUP / DEFINITION
# 8. ADD NODES
#
# We have three Nodes:
#
# 1. LLM Node
# 2. Tool Node
# 3. Final LLM Node
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
    "final_llm_node",
    final_llm_node
)


# ============================================================
# SETUP / DEFINITION
# 9. ADD EDGES
#
# Edges define the execution path.
#
# START
#   ↓
# LLM Node
#   ↓
# Tool Node
#   ↓
# Final LLM Node
#   ↓
# END
# ============================================================

graph.add_edge(
    START,
    "llm_node"
)

graph.add_edge(
    "llm_node",
    "tool_node"
)

graph.add_edge(
    "tool_node",
    "final_llm_node"
)

graph.add_edge(
    "final_llm_node",
    END
)


# ============================================================
# SETUP / DEFINITION
# 10. COMPILE GRAPH
# ============================================================

app = graph.compile()


# ============================================================
# RUNTIME FLOW
# 11. GET USER QUESTION
# ============================================================

question = input(
    "\nAsk a question: "
)


# ============================================================
# RUNTIME FLOW
# 12. INITIAL STATE
# ============================================================

initial_state = {
    "question": question,
    "tool_result": "",
    "answer": ""
}


# ============================================================
# RUNTIME FLOW
# 13. EXECUTE GRAPH
#
# START
#   ↓
# LLM Node
#   ↓
# Tool Node
#   ↓
# Final LLM Node
#   ↓
# END
# ============================================================

result = app.invoke(
    initial_state
)


# ============================================================
# RUNTIME FLOW
# 14. FINAL STATE
# ============================================================

print("\nFINAL STATE:")

print(result)


# ============================================================
# RUNTIME FLOW
# 15. FINAL ANSWER
# ============================================================

print("\nFINAL ANSWER:")

print(result["answer"])