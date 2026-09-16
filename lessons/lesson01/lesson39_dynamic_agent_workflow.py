# ============================================================
# LESSON 39 — DYNAMIC MULTI-STEP AGENT WORKFLOW
#
# Goal:
# Understand how an Agent can dynamically decide what to do
# next based on the current State.
#
#
# LESSON 38:
#
# Fixed workflow:
#
# User
#   ↓
# Memory
#   ↓
# RAG
#   ↓
# Final LLM
#
#
# LESSON 39:
#
# Dynamic workflow:
#
# User
#   ↓
# LLM decides
#   ↓
# ┌───────────────┐
# │ What next?    │
# └───────┬───────┘
#         │
#     ┌───┴────┐
#     ↓        ↓
#  Memory     RAG
#     │        │
#     └───┬────┘
#         ↓
#      LLM decides again
#         ↓
#   More work needed?
#     │          │
#    YES         NO
#     ↓          ↓
#   Tool       Final
#
#
# KEY CONCEPT:
#
# The Agent is NOT following a fixed sequence.
#
# The LLM looks at the current State and decides what should
# happen next.
#
#
# This is the beginning of an AGENT LOOP.
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
# question      → user's request
# memory_result → information retrieved from memory
# rag_result    → information retrieved from enterprise
#                 knowledge
# next_action   → what the Agent should do next
# answer        → final answer
# ============================================================

class AgentState(TypedDict):

    question: str
    memory_result: str
    rag_result: str
    next_action: str
    answer: str


# ============================================================
# SETUP / DEFINITION
# 4. MEMORY TOOL
#
# This represents the Agent's long-term user memory.
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
# This represents enterprise knowledge retrieval.
# ============================================================

def rag_tool(state: AgentState):

    print("\n----------------------------------------")
    print("RAG TOOL NODE")
    print("----------------------------------------")

    result = (
        "Employees receive 24 days of annual leave "
        "per year."
    )

    print("RAG result:")
    print(result)

    return {
        "rag_result": result
    }


# ============================================================
# SETUP / DEFINITION
# 6. DECISION NODE
#
# This is the most important Node in this lesson.
#
# The LLM looks at:
#
#   User question
#   What memory has already been retrieved
#   What RAG information has already been retrieved
#
# and decides what should happen NEXT.
#
#
# Possible decisions:
#
#   memory
#   rag
#   final
#
#
# Notice:
#
# The decision can change after each Tool execution.
# ============================================================

def decision_node(state: AgentState):

    print("\n========================================")
    print("LLM DECISION NODE")
    print("========================================")

    prompt = f"""
You are the decision-making part of an Agent.

User question:
{state["question"]}

Memory already retrieved:
{state["memory_result"]}

Enterprise information already retrieved:
{state["rag_result"]}

Decide what the Agent should do NEXT.

Choose exactly ONE:

memory
→ if user memory is needed and has not yet been retrieved.

rag
→ if enterprise/company information is needed and has
  not yet been retrieved.

final
→ if enough information has already been collected and
  the Agent can answer the user.

Important:
Do not select memory if memory_result is already populated.
Do not select rag if rag_result is already populated.

Respond with only:

memory

or:

rag

or:

final
"""

    response = llm.invoke(prompt)

    next_action = response.content.strip().lower()

    if next_action not in [
        "memory",
        "rag",
        "final"
    ]:

        next_action = "final"

    print("Next action selected by LLM:")
    print(next_action)

    return {
        "next_action": next_action
    }


# ============================================================
# SETUP / DEFINITION
# 7. ROUTING FUNCTION
#
# LangGraph reads the LLM's decision from State and chooses
# the next Node.
# ============================================================

def route_next(state: AgentState):

    if state["next_action"] == "memory":

        return "memory"

    if state["next_action"] == "rag":

        return "rag"

    return "final"


# ============================================================
# SETUP / DEFINITION
# 8. FINAL LLM NODE
#
# Once the Agent decides that enough information has been
# collected, the final LLM generates the answer.
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

Use the available information to answer the question.

Do not invent information.

Give a short, natural answer.
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


# ============================================================
# SETUP / DEFINITION
# 9. CREATE LANGGRAPH
#
# LangGraph acts as the ORCHESTRATOR.
#
# It will repeatedly:
#
#   1. Run the decision Node
#   2. Follow the selected Edge
#   3. Run the selected capability
#   4. Return to the decision Node
#
# until the LLM chooses "final".
# ============================================================

graph = StateGraph(AgentState)


# ============================================================
# SETUP / DEFINITION
# 10. ADD NODES
# ============================================================

graph.add_node(
    "decision_node",
    decision_node
)

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
# 11. START → DECISION
# ============================================================

graph.add_edge(
    START,
    "decision_node"
)


# ============================================================
# SETUP / DEFINITION
# 12. CONDITIONAL EDGE
#
# The LLM decides which capability should run next.
#
#
#                  DECISION
#                     ↓
#          ┌──────────┼──────────┐
#          ↓          ↓          ↓
#       MEMORY       RAG       FINAL
#          ↓          ↓          ↓
#          └────┬─────┘          │
#               ↓                │
#          DECISION AGAIN        │
#                                ↓
#                              END
#
#
# This return to the decision Node creates the Agent loop.
# ============================================================

graph.add_conditional_edges(
    "decision_node",
    route_next,
    {
        "memory": "memory_tool",
        "rag": "rag_tool",
        "final": "final_llm_node"
    }
)


# ============================================================
# SETUP / DEFINITION
# 13. MEMORY → DECISION
#
# After Memory finishes, the Agent evaluates the updated
# State again.
# ============================================================

graph.add_edge(
    "memory_tool",
    "decision_node"
)


# ============================================================
# SETUP / DEFINITION
# 14. RAG → DECISION
#
# After RAG finishes, the Agent evaluates the updated
# State again.
# ============================================================

graph.add_edge(
    "rag_tool",
    "decision_node"
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

    "memory_result": "",

    "rag_result": "",

    "next_action": "",

    "answer": ""
}


# ============================================================
# RUNTIME FLOW
# 19. SHOW INITIAL STATE
# ============================================================

print("\n========================================")
print("INITIAL STATE")
print("========================================")

print(initial_state)


# ============================================================
# RUNTIME FLOW
# 20. RUN LANGGRAPH ORCHESTRATOR
#
# LangGraph now executes a DYNAMIC workflow.
#
# The LLM can choose:
#
#   Memory
#   RAG
#   Final
#
# After Memory or RAG, execution returns to the Decision Node.
#
# ============================================================

print("\n========================================")
print("LANGGRAPH ORCHESTRATOR")
print("========================================")

result = app.invoke(
    initial_state
)


# ============================================================
# RUNTIME FLOW
# 21. FINAL STATE
# ============================================================

print("\n========================================")
print("FINAL STATE")
print("========================================")

print(result)


# ============================================================
# RUNTIME FLOW
# 22. FINAL ANSWER
# ============================================================

print("\n========================================")
print("FINAL ANSWER")
print("========================================")

print(result["answer"])