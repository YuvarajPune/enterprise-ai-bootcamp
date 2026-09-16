# ============================================================
# LESSON 34 — MEMORY RETRIEVAL AS A TOOL
#
# Goal:
# Let the LLM decide when it needs to retrieve information
# from semantic memory.
#
# LESSON 33:
#
# User
#   ↓
# Application retrieves memory
#   ↓
# LLM receives memory
#   ↓
# Answer
#
#
# LESSON 34:
#
# User
#   ↓
# LLM
#   ↓
# "Do I need memory?"
#   ↓
# Memory Tool
#   ↓
# SQLite Database
#   ↓
# Memory Result
#   ↓
# LLM
#   ↓
# Final Answer
#
#
# KEY CONCEPT:
#
# The LLM does NOT directly access SQLite.
#
# The LLM decides:
#   "I need memory."
#
# The Agent application then executes the Memory Tool.
#
#
# RUNTIME ARCHITECTURE:
#
#                         ┌──→ FINAL
#                         │
# User → LLM → Decision ──┤
#                         │
#                         └──→ MEMORY TOOL
#                                  ↓
#                              SQLite DB
#                                  ↓
#                               Memory
#                                  ↓
#                                 LLM
#                                  ↓
#                                FINAL
# ============================================================


# ============================================================
# SETUP / DEFINITION
# 1. IMPORTS
# ============================================================

import sqlite3

from typing import TypedDict

from langchain_groq import ChatGroq
from langchain_core.tools import tool


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
# 3. SQLITE DATABASE
#
# This is the persistent semantic memory database created
# in Lesson 33.
#
# IMPORTANT:
# We are reusing the existing database.
#
# Do NOT create a new memory database for this lesson.
# ============================================================

connection = sqlite3.connect(
    "semantic_memory.db"
)


# ============================================================
# SETUP / DEFINITION
# 4. STATE
#
# State carries information through the workflow.
#
# question      → user's question
# decision      → LLM's decision
# memory        → retrieved memory
# answer        → final answer
# ============================================================

class AgentState(TypedDict):

    question: str
    decision: str
    memory: str
    answer: str


# ============================================================
# SETUP / DEFINITION
# 5. MEMORY RETRIEVAL TOOL
#
# This is the important new component.
#
# The LLM does not directly access SQLite.
#
# Instead:
#
# LLM
#   ↓
# decides it needs memory
#   ↓
# Application executes this Tool
#   ↓
# SQLite
#   ↓
# Memory returned
# ============================================================

@tool
def retrieve_memory() -> str:
    """
    Retrieve stored semantic memory from SQLite.
    """

    cursor = connection.execute(
        """
        SELECT key, value
        FROM semantic_memory
        """
    )

    rows = cursor.fetchall()

    if not rows:

        return "No semantic memory is stored."

    memories = []

    for key, value in rows:

        memories.append(
            f"{key} = {value}"
        )

    return "\n".join(memories)


# ============================================================
# SETUP / DEFINITION
# 6. LLM DECISION NODE
#
# The LLM decides whether the current question requires
# stored memory.
#
# Possible decisions:
#
#   memory → use Memory Tool
#   final  → answer directly
# ============================================================

def llm_decision_node(state: AgentState):

    print("\nLLM DECISION NODE")

    prompt = f"""
You are an Agent deciding whether stored user memory
is relevant to a question.

User question:
{state["question"]}

Stored memory may contain user preferences or facts.

If the question requires a stored user preference or fact,
respond with exactly:

memory

Otherwise respond with exactly:

final

Do not provide any other text.
"""

    response = llm.invoke(prompt)

    decision = response.content.strip().lower()

    if decision not in ["memory", "final"]:

        decision = "final"

    print("LLM decision:", decision)

    return {
        "decision": decision
    }


# ============================================================
# SETUP / DEFINITION
# 7. ROUTING FUNCTION
#
# This Conditional Edge reads the decision made by the LLM.
#
# memory → Memory Tool
# final  → Final LLM
# ============================================================

def route_after_decision(state: AgentState):

    if state["decision"] == "memory":

        return "memory"

    return "final"


# ============================================================
# SETUP / DEFINITION
# 8. MEMORY TOOL NODE
#
# The application executes the Memory Tool here.
#
# This is where the actual database access happens.
# ============================================================

def memory_tool_node(state: AgentState):

    print("\nMEMORY TOOL NODE: Retrieving memory")

    memory_result = retrieve_memory.invoke({})

    print("Retrieved memory:")

    print(memory_result)

    return {
        "memory": memory_result
    }


# ============================================================
# SETUP / DEFINITION
# 9. FINAL LLM NODE
#
# The LLM now receives:
#
#   User question
#   Retrieved memory
#
# and produces the final answer.
# ============================================================

def final_llm_node(state: AgentState):

    print("\nFINAL LLM NODE: Preparing answer")

    prompt = f"""
You are a helpful assistant.

User question:
{state["question"]}

Relevant remembered information:
{state["memory"]}

Use the remembered information when it is relevant.

If there is no relevant memory, answer normally.

Give a short and natural answer.
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


# ============================================================
# SETUP / DEFINITION
# 10. CREATE GRAPH
# ============================================================

from langgraph.graph import StateGraph, START, END

graph = StateGraph(AgentState)


# ============================================================
# SETUP / DEFINITION
# 11. ADD NODES
#
# Nodes:
#
#   1. LLM Decision Node
#   2. Memory Tool Node
#   3. Final LLM Node
# ============================================================

graph.add_node(
    "llm_decision_node",
    llm_decision_node
)

graph.add_node(
    "memory_tool_node",
    memory_tool_node
)

graph.add_node(
    "final_llm_node",
    final_llm_node
)


# ============================================================
# SETUP / DEFINITION
# 12. START → LLM DECISION
# ============================================================

graph.add_edge(
    START,
    "llm_decision_node"
)


# ============================================================
# SETUP / DEFINITION
# 13. CONDITIONAL EDGE
#
# LLM Decision
#       ↓
# ┌─────┴─────┐
# ↓           ↓
# Memory     Final
# Tool
# ============================================================

graph.add_conditional_edges(
    "llm_decision_node",
    route_after_decision,
    {
        "memory": "memory_tool_node",
        "final": "final_llm_node"
    }
)


# ============================================================
# SETUP / DEFINITION
# 14. MEMORY TOOL → FINAL LLM
#
# Retrieved memory is placed into State.
#
# Then the LLM uses that memory.
# ============================================================

graph.add_edge(
    "memory_tool_node",
    "final_llm_node"
)


# ============================================================
# SETUP / DEFINITION
# 15. FINAL LLM → END
# ============================================================

graph.add_edge(
    "final_llm_node",
    END
)


# ============================================================
# SETUP / DEFINITION
# 16. COMPILE GRAPH
# ============================================================

app = graph.compile()


# ============================================================
# RUNTIME FLOW
# 17. GET USER QUESTION
# ============================================================

question = input(
    "\nAsk a question: "
)


# ============================================================
# RUNTIME FLOW
# 18. INITIAL STATE
# ============================================================

initial_state = {
    "question": question,
    "decision": "",
    "memory": "",
    "answer": ""
}


# ============================================================
# RUNTIME FLOW
# 19. EXECUTE GRAPH
#
#
#                         ┌──→ FINAL
#                         │
# User → LLM → Decision ──┤
#                         │
#                         └──→ Memory Tool
#                                  ↓
#                              SQLite DB
#                                  ↓
#                               Memory
#                                  ↓
#                                 LLM
#                                  ↓
#                                FINAL
# ============================================================

result = app.invoke(
    initial_state
)


# ============================================================
# RUNTIME FLOW
# 20. FINAL STATE
# ============================================================

print("\nFINAL STATE:")

print(result)


# ============================================================
# RUNTIME FLOW
# 21. FINAL ANSWER
# ============================================================

print("\nFINAL ANSWER:")

print(result["answer"])


# ============================================================
# RUNTIME FLOW
# 22. CLOSE DATABASE CONNECTION
# ============================================================

connection.close()