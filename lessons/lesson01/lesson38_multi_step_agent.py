# ============================================================
# LESSON 38 — MULTI-STEP AGENT WORKFLOW
#
# Goal:
# Understand how an Agent can use MORE THAN ONE capability
# in sequence.
#
#
# LESSON 37:
#
# User
#   ↓
# LLM Decision
#   ↓
# ONE capability
#   ↓
# Final LLM
#   ↓
# Answer
#
#
# LESSON 38:
#
# User
#   ↓
# LLM
#   ↓
# Memory Tool
#   ↓
# RAG Tool
#   ↓
# Final LLM
#   ↓
# Answer
#
#
# KEY CONCEPT:
#
# A multi-step Agent can:
#
#   1. Retrieve user information
#   2. Retrieve enterprise information
#   3. Combine both
#   4. Ask the LLM to produce the final answer
#
#
# IMPORTANT:
#
# State is accumulated as the workflow progresses.
#
# Initial State
#      ↓
# Memory result added
#      ↓
# RAG result added
#      ↓
# Final LLM
#      ↓
# Final State
# ============================================================


# ============================================================
# SETUP / DEFINITION
# 1. IMPORTS
# ============================================================

from typing import TypedDict

from langchain_groq import ChatGroq

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
# 3. STATE
#
# State is the shared information carried through the
# workflow.
#
# question       → user's request
# memory_result  → information about the user
# rag_result     → enterprise information
# answer         → final response
# ============================================================

class AgentState(TypedDict):

    question: str
    memory_result: str
    rag_result: str
    answer: str


# ============================================================
# SETUP / DEFINITION
# 4. MEMORY TOOL
#
# In a production Agent, this could retrieve information
# from SQL or Vector DB.
#
# For this lesson we use a simple fixed memory so that
# we can focus on orchestration.
# ============================================================

def memory_tool(state: AgentState):

    print("\n----------------------------------------")
    print("MEMORY TOOL NODE")
    print("----------------------------------------")

    result = (
        "User prefers Excel reports and concise "
        "executive summaries."
    )

    print("Memory result:")
    print(result)

    return {
        "memory_result": result
    }


# ============================================================
# SETUP / DEFINITION
# 5. RAG TOOL
#
# In a production Agent, this could search:
#
#   Vector DB
#   Enterprise documents
#   Knowledge base
#
# For this lesson we use a simple example.
# ============================================================

def rag_tool(state: AgentState):

    print("\n----------------------------------------")
    print("RAG TOOL NODE")
    print("----------------------------------------")

    question = state["question"].lower()

    if "leave" in question:

        result = (
            "Employees receive 24 days of annual leave "
            "per year."
        )

    else:

        result = (
            "No relevant enterprise information was found."
        )

    print("RAG result:")
    print(result)

    return {
        "rag_result": result
    }


# ============================================================
# SETUP / DEFINITION
# 6. FINAL LLM NODE
#
# The LLM receives information gathered by BOTH previous
# steps.
#
# Memory:
#     User preferences
#
# RAG:
#     Enterprise knowledge
#
# The LLM combines them to produce the final answer.
# ============================================================

def final_llm_node(state: AgentState):

    print("\n----------------------------------------")
    print("FINAL LLM NODE")
    print("----------------------------------------")

    prompt = f"""
You are a helpful enterprise assistant.

User question:
{state["question"]}

User memory:
{state["memory_result"]}

Enterprise information:
{state["rag_result"]}

Use the information above to answer the user's question.

Do not invent information.

Give a short, natural answer.
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


# ============================================================
# SETUP / DEFINITION
# 7. CREATE LANGGRAPH
#
# LangGraph will orchestrate the multi-step workflow.
# ============================================================

graph = StateGraph(AgentState)


# ============================================================
# SETUP / DEFINITION
# 8. ADD NODES
#
# Nodes:
#
#   memory_tool
#   rag_tool
#   final_llm_node
# ============================================================

graph.add_node(
    "memory_tool",
    memory_tool
)

graph.add_node(
    "rag_tool",
    rag_tool
)

graph.add_node(
    "final_llm_node",
    final_llm_node
)


# ============================================================
# SETUP / DEFINITION
# 9. DEFINE WORKFLOW
#
# START
#   ↓
# Memory Tool
#   ↓
# RAG Tool
#   ↓
# Final LLM
#   ↓
# END
# ============================================================

graph.add_edge(
    START,
    "memory_tool"
)

graph.add_edge(
    "memory_tool",
    "rag_tool"
)

graph.add_edge(
    "rag_tool",
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
#
# At the beginning, only the user's question is available.
#
# Memory and RAG results are empty.
# ============================================================

initial_state = {

    "question": question,

    "memory_result": "",

    "rag_result": "",

    "answer": ""
}


# ============================================================
# RUNTIME FLOW
# 13. SHOW INITIAL STATE
# ============================================================

print("\n========================================")
print("INITIAL STATE")
print("========================================")

print(initial_state)


# ============================================================
# RUNTIME FLOW
# 14. RUN LANGGRAPH ORCHESTRATOR
#
# LangGraph now executes the nodes in sequence:
#
#   Memory
#      ↓
#   RAG
#      ↓
#   Final LLM
#
# State is updated after each Node.
# ============================================================

print("\n========================================")
print("LANGGRAPH ORCHESTRATOR")
print("========================================")

result = app.invoke(
    initial_state
)


# ============================================================
# RUNTIME FLOW
# 15. FINAL STATE
#
# By now State contains:
#
#   question
#   memory_result
#   rag_result
#   answer
# ============================================================

print("\n========================================")
print("FINAL STATE")
print("========================================")

print(result)


# ============================================================
# RUNTIME FLOW
# 16. FINAL ANSWER
# ============================================================

print("\n========================================")
print("FINAL ANSWER")
print("========================================")

print(result["answer"])