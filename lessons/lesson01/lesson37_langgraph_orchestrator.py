# ============================================================
# LESSON 37 — LANGGRAPH AS AN ORCHESTRATOR
#
# Goal:
# Understand how LangGraph orchestrates multiple components
# of an Agent workflow.
#
#
# We already learned these components individually:
#
#   LLM
#   Tools
#   RAG
#   Memory
#   State
#   Conditional Edges
#
# Now we combine them.
#
#
# RUNTIME ARCHITECTURE:
#
#                         ┌──→ RAG TOOL ────┐
#                         │                 │
# User → LLM → Decision ──┼──→ MEMORY TOOL ┤
#                         │                 │
#                         └──→ DIRECT ──────┤
#                                           ↓
#                                          LLM
#                                           ↓
#                                         FINAL
#
#
# KEY CONCEPT:
#
# LangGraph is acting as the ORCHESTRATOR.
#
# It controls:
#
#   Which Node runs
#   Which Tool is used
#   What State is passed
#   Which path is followed
#   When the workflow ends
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
# State is the shared information flowing through the graph.
#
# question       → user's question
# decision       → LLM's routing decision
# tool_result    → result from RAG or Memory Tool
# answer         → final LLM answer
# ============================================================

class AgentState(TypedDict):

    question: str
    decision: str
    tool_result: str
    answer: str


# ============================================================
# SETUP / DEFINITION
# 4. RAG TOOL
#
# In a real Agent this could connect to:
#
#   Vector DB
#   Document store
#   Enterprise search
#   RAG pipeline
#
# For this orchestration lesson, we use a simple function.
# ============================================================

def rag_tool(state: AgentState):

    print("\nRAG TOOL: Searching enterprise knowledge")

    question = state["question"].lower()

    if "leave" in question:

        result = (
            "Employees receive 24 days of annual leave "
            "per year."
        )

    elif "work from home" in question:

        result = (
            "Employees can work from home up to "
            "two days per week."
        )

    else:

        result = (
            "No relevant enterprise knowledge was found."
        )

    print("RAG result:", result)

    return {
        "tool_result": result
    }


# ============================================================
# SETUP / DEFINITION
# 5. MEMORY TOOL
#
# In a real system this could retrieve:
#
#   User preferences
#   Previous interactions
#   Long-term semantic memory
#   Customer information
#
# For this lesson we use a simple example.
# ============================================================

def memory_tool(state: AgentState):

    print("\nMEMORY TOOL: Retrieving user memory")

    result = (
        "User prefers concise executive summaries "
        "and Excel reports."
    )

    print("Memory result:", result)

    return {
        "tool_result": result
    }


# ============================================================
# SETUP / DEFINITION
# 6. LLM DECISION NODE
#
# The LLM decides which capability is required.
#
# Possible decisions:
#
#   rag
#   memory
#   direct
# ============================================================

def decision_node(state: AgentState):

    print("\nLLM DECISION NODE: Thinking")

    prompt = f"""
You are an Agent deciding which capability is required.

User question:
{state["question"]}

Choose exactly ONE:

rag
memory
direct

Use:

rag
→ when the question asks about company policies,
  procedures, or enterprise knowledge.

memory
→ when the question asks about the user's preferences
  or previously remembered information.

direct
→ when no external information is required.

Respond with only:
rag
memory
or
direct
"""

    response = llm.invoke(prompt)

    decision = response.content.strip().lower()

    if decision not in [
        "rag",
        "memory",
        "direct"
    ]:

        decision = "direct"

    print("LLM decision:", decision)

    return {
        "decision": decision
    }


# ============================================================
# SETUP / DEFINITION
# 7. ROUTING FUNCTION
#
# This function reads the decision from State.
#
# LangGraph uses the result to select the next Node.
# ============================================================

def route_after_decision(state: AgentState):

    if state["decision"] == "rag":

        return "rag"

    if state["decision"] == "memory":

        return "memory"

    return "direct"


# ============================================================
# SETUP / DEFINITION
# 8. DIRECT PATH
#
# This path is used when no Tool is required.
# ============================================================

def direct_node(state: AgentState):

    print("\nDIRECT NODE: No external capability required")

    return {
        "tool_result": "No external information was required."
    }


# ============================================================
# SETUP / DEFINITION
# 9. FINAL LLM NODE
#
# The final LLM receives:
#
#   User question
#   Tool result
#
# and produces the final answer.
# ============================================================

def final_llm_node(state: AgentState):

    print("\nFINAL LLM NODE: Preparing answer")

    prompt = f"""
You are a helpful assistant.

User question:
{state["question"]}

Information available to you:
{state["tool_result"]}

Use the information when relevant.

If the information says that nothing external was required,
answer the question normally.

Do not invent information.

Give a short, natural answer.
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
#
# Nodes:
#
#   decision_node
#   rag_tool
#   memory_tool
#   direct_node
#   final_llm_node
# ============================================================

graph.add_node(
    "decision_node",
    decision_node
)

graph.add_node(
    "rag_tool",
    rag_tool
)

graph.add_node(
    "memory_tool",
    memory_tool
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
# 12. START → DECISION NODE
# ============================================================

graph.add_edge(
    START,
    "decision_node"
)


# ============================================================
# SETUP / DEFINITION
# 13. CONDITIONAL EDGE
#
# The LLM decision determines which capability is used.
#
#
#                     decision
#                        ↓
#              ┌─────────┼─────────┐
#              ↓         ↓         ↓
#             RAG      Memory     Direct
#              ↓         ↓         ↓
#              └─────────┼─────────┘
#                        ↓
#                    Final LLM
# ============================================================

graph.add_conditional_edges(
    "decision_node",
    route_after_decision,
    {
        "rag": "rag_tool",
        "memory": "memory_tool",
        "direct": "direct_node"
    }
)


# ============================================================
# SETUP / DEFINITION
# 14. TOOL NODES → FINAL LLM
#
# Whichever capability was selected, its result goes into
# State and then to the Final LLM.
# ============================================================

graph.add_edge(
    "rag_tool",
    "final_llm_node"
)

graph.add_edge(
    "memory_tool",
    "final_llm_node"
)

graph.add_edge(
    "direct_node",
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

    "tool_result": "",

    "answer": ""
}


# ============================================================
# RUNTIME FLOW
# 19. EXECUTE GRAPH
#
#
#                     LangGraph
#                     Orchestrator
#                          ↓
#                       Decision
#                          ↓
#              ┌───────────┼───────────┐
#              ↓           ↓           ↓
#             RAG        Memory       Direct
#              ↓           ↓           ↓
#              └───────────┼───────────┘
#                          ↓
#                      Final LLM
#                          ↓
#                         END
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