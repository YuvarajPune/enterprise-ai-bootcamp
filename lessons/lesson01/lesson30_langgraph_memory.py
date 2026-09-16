# ============================================================
# LESSON 30 — LANGGRAPH MEMORY + LLM
#
# Goal:
# Understand how an LLM can use information retained by
# LangGraph across multiple interactions.
#
# KEY DISTINCTION:
#
# State:
#   Information available during the current workflow.
#
# Memory / Checkpoint:
#   State saved so it can be restored in a later interaction.
#
#
# RUNTIME ARCHITECTURE:
#
# Conversation 1
#       ↓
#      LLM
#       ↓
#     State
#       ↓
#  Checkpointer
#       ↓
#  Saved State
#       ↓
# Conversation 2
#       ↓
# Restore State
#       ↓
#      LLM
#       ↓
#     Answer
# ============================================================


# ============================================================
# SETUP / DEFINITION
# 1. IMPORTS
# ============================================================

from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

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
# The State is the information carried through the graph.
#
# user_name → information we want to remember
# message   → current user message
# answer    → LLM response
# ============================================================

class AgentState(TypedDict):

    user_name: str
    message: str
    answer: str


# ============================================================
# SETUP / DEFINITION
# 4. MEMORY / CHECKPOINTER
#
# MemorySaver saves the State associated with a thread.
#
# IMPORTANT:
# This memory exists only while this Python program is
# running.
# ============================================================

memory = MemorySaver()


# ============================================================
# SETUP / DEFINITION
# 5. LLM NODE
#
# The LLM receives:
#
#   1. Current message
#   2. Information already stored in State
#
# The LLM can therefore use the remembered information.
# ============================================================

def llm_node(state: AgentState):

    print("\nLLM NODE: Processing message")

    prompt = f"""
You are a helpful assistant.

Previously remembered user name:
{state["user_name"]}

Current user message:
{state["message"]}

If the user tells you their name, remember it.

Answer the user's message naturally.

If the user asks for their name and a name is remembered,
use that name.
"""

    response = llm.invoke(prompt)

    # --------------------------------------------------------
    # Detect a simple name statement for this demonstration.
    #
    # This is NOT the LLM memory mechanism.
    # It simply lets us place the name into State.
    # --------------------------------------------------------

    user_name = state["user_name"]

    message_lower = state["message"].lower()

    if "my name is" in message_lower:

        name = (
            message_lower
            .split("my name is", 1)[1]
            .strip()
        )

        user_name = name.capitalize()

    return {
        "user_name": user_name,
        "answer": response.content
    }


# ============================================================
# SETUP / DEFINITION
# 6. CREATE THE GRAPH
# ============================================================

graph = StateGraph(AgentState)


# ============================================================
# SETUP / DEFINITION
# 7. ADD LLM NODE
# ============================================================

graph.add_node(
    "llm_node",
    llm_node
)


# ============================================================
# SETUP / DEFINITION
# 8. ADD EDGES
#
# START
#   ↓
# LLM Node
#   ↓
# END
# ============================================================

graph.add_edge(
    START,
    "llm_node"
)

graph.add_edge(
    "llm_node",
    END
)


# ============================================================
# SETUP / DEFINITION
# 9. COMPILE GRAPH WITH MEMORY
#
# The checkpointer enables State to be saved and restored.
# ============================================================

app = graph.compile(
    checkpointer=memory
)


# ============================================================
# RUNTIME FLOW
# 10. THREAD ID
#
# thread_id identifies this conversation.
#
# Using the same thread_id allows LangGraph to restore the
# State from the previous interaction.
# ============================================================

config = {
    "configurable": {
        "thread_id": "user_raj"
    }
}


# ============================================================
# RUNTIME FLOW
# 11. CONVERSATION 1
# ============================================================

message_1 = input(
    "\nConversation 1 — Enter a message: "
)


# ============================================================
# RUNTIME FLOW
# 12. INITIAL STATE
# ============================================================

initial_state = {
    "user_name": "",
    "message": message_1,
    "answer": ""
}


# ============================================================
# RUNTIME FLOW
# 13. FIRST EXECUTION
#
# User
#   ↓
# State
#   ↓
# LLM
#   ↓
# Updated State
#   ↓
# Checkpoint
# ============================================================

result_1 = app.invoke(
    initial_state,
    config
)


# ============================================================
# RUNTIME FLOW
# 14. FINAL STATE — CONVERSATION 1
# ============================================================

print("\nFINAL STATE — CONVERSATION 1:")

print(result_1)


print("\nFINAL ANSWER — CONVERSATION 1:")

print(result_1["answer"])


# ============================================================
# RUNTIME FLOW
# 15. CONVERSATION 2
#
# IMPORTANT:
# We use the SAME thread_id.
#
# LangGraph can therefore restore the previous State.
# ============================================================

message_2 = input(
    "\nConversation 2 — Enter a message: "
)


# ============================================================
# RUNTIME FLOW
# 16. SECOND EXECUTION
#
# We provide the new message.
#
# LangGraph restores the previous State for this thread.
# ============================================================

result_2 = app.invoke(
    {
        "message": message_2,
        "answer": ""
    },
    config
)


# ============================================================
# RUNTIME FLOW
# 17. FINAL STATE — CONVERSATION 2
# ============================================================

print("\nFINAL STATE — CONVERSATION 2:")

print(result_2)


# ============================================================
# RUNTIME FLOW
# 18. FINAL ANSWER — CONVERSATION 2
# ============================================================

print("\nFINAL ANSWER — CONVERSATION 2:")

print(result_2["answer"])