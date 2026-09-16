# ============================================================
# LESSON 35 — TARGETED MEMORY RETRIEVAL
#
# Goal:
# Make the Memory Tool retrieve only the specific memory
# requested by the Agent.
#
# LESSON 34:
#
# LLM
#   ↓
# "I need memory"
#   ↓
# Memory Tool
#   ↓
# SQLite
#   ↓
# ALL memories
#
#
# LESSON 35:
#
# LLM
#   ↓
# "I need report_format"
#   ↓
# Memory Tool(report_format)
#   ↓
# SQLite
#   ↓
# ONLY report_format
#   ↓
# LLM
#   ↓
# Final Answer
#
#
# KEY CONCEPT:
#
# Targeted retrieval = retrieve only the memory relevant
# to the current task.
#
# This is structured SQL lookup.
# Later we will compare this with semantic/vector retrieval.
# ============================================================


# ============================================================
# SETUP / DEFINITION
# 1. IMPORTS
# ============================================================

import sqlite3

from typing import TypedDict

from langchain_groq import ChatGroq
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END


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
# We reuse the persistent semantic memory database created
# in Lesson 33.
#
# Existing database:
#
#     semantic_memory.db
#
# We are NOT creating a new memory database.
# ============================================================

connection = sqlite3.connect(
    "semantic_memory.db"
)


# ============================================================
# SETUP / DEFINITION
# 4. STATE
#
# State carries information through the graph.
#
# question      → user's question
# memory_key    → specific memory requested
# decision      → LLM's decision
# memory        → retrieved memory
# answer        → final answer
# ============================================================

class AgentState(TypedDict):

    question: str
    memory_key: str
    decision: str
    memory: str
    answer: str


# ============================================================
# SETUP / DEFINITION
# 5. MEMORY RETRIEVAL TOOL
#
# This Tool receives a SPECIFIC memory key.
#
# Example:
#
#     report_format
#
# It then retrieves ONLY that memory from SQLite.
# ============================================================

@tool
def retrieve_memory(memory_key: str) -> str:
    """
    Retrieve one specific semantic memory from SQLite.
    """

    cursor = connection.execute(
        """
        SELECT key, value
        FROM semantic_memory
        WHERE key = ?
        """,
        (memory_key,)
    )

    row = cursor.fetchone()

    if row is None:

        return (
            f"No memory found for: {memory_key}"
        )

    key, value = row

    return f"{key} = {value}"


# ============================================================
# SETUP / DEFINITION
# 6. LLM DECISION NODE
#
# The LLM decides:
#
#   memory → memory is required
#   final  → memory is not required
#
# If memory is required, the LLM also identifies WHICH
# memory it needs.
# ============================================================

def llm_decision_node(state: AgentState):

    print("\nLLM DECISION NODE")

    prompt = f"""
You are an Agent deciding whether stored user memory
is relevant to a question.

User question:
{state["question"]}

Possible stored memory keys include:

report_format
favorite_language
department

If a stored user preference or fact is required,
respond exactly in this format:

memory|KEY

For example:

memory|report_format

If memory is not required, respond:

final

Do not provide any other text.
"""

    response = llm.invoke(prompt)

    decision = response.content.strip().lower()

    # --------------------------------------------------------
    # Validate the LLM output.
    # --------------------------------------------------------

    if decision.startswith("memory|"):

        memory_key = decision.split(
            "|", 1
        )[1].strip()

        return {
            "decision": "memory",
            "memory_key": memory_key
        }

    return {
        "decision": "final",
        "memory_key": ""
    }


# ============================================================
# SETUP / DEFINITION
# 7. CONDITIONAL ROUTING FUNCTION
#
# This reads the LLM's decision from State.
# ============================================================

def route_after_decision(state: AgentState):

    if state["decision"] == "memory":

        return "memory"

    return "final"


# ============================================================
# SETUP / DEFINITION
# 8. MEMORY TOOL NODE
#
# The application executes the Memory Tool using the
# specific memory_key selected by the LLM.
#
# LLM:
#     memory|report_format
#
# Application:
#     retrieve_memory("report_format")
# ============================================================

def memory_tool_node(state: AgentState):

    print("\nMEMORY TOOL NODE")

    print(
        "Requested memory:",
        state["memory_key"]
    )

    memory_result = retrieve_memory.invoke(
        {
            "memory_key": state["memory_key"]
        }
    )

    print(
        "Retrieved memory:",
        memory_result
    )

    return {
        "memory": memory_result
    }


# ============================================================
# SETUP / DEFINITION
# 9. FINAL LLM NODE
#
# The LLM receives:
#
#   User question
#   Targeted memory
#
# and prepares the final answer.
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

graph = StateGraph(AgentState)


# ============================================================
# SETUP / DEFINITION
# 11. ADD NODES
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
# LLM
#  ↓
# ┌─────────────┐
# │             │
# memory       final
# │             │
# ↓             ↓
# Memory       Final LLM
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
    "memory_key": "",
    "decision": "",
    "memory": "",
    "answer": ""
}


# ============================================================
# RUNTIME FLOW
# 19. EXECUTE GRAPH
#
#
# User
#  ↓
# LLM
#  ↓
# memory|report_format
#  ↓
# Conditional Edge
#  ↓
# Memory Tool(report_format)
#  ↓
# SQLite
#  ↓
# report_format = Excel
#  ↓
# Final LLM
#  ↓
# END
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