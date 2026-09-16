# ============================================================
# LESSON 31 — LANGGRAPH PERSISTENT MEMORY
#
# Goal:
# Understand how LangGraph can save State permanently so
# that it survives after the Python program is stopped.
#
# Lesson 30:
#   MemorySaver → RAM → lost when program stops
#
# Lesson 31:
#   SQLite Checkpointer → database file → persists
#
#
# RUNTIME ARCHITECTURE:
#
# User
#   ↓
# LangGraph
#   ↓
# LLM Node
#   ↓
# State
#   ↓
# SQLite Checkpointer
#   ↓
# memory.db
#
# After program stops:
#
# memory.db remains
#       ↓
# Program starts again
#       ↓
# Same thread_id
#       ↓
# Previous State restored
# ============================================================


# ============================================================
# SETUP / DEFINITION
# 1. IMPORTS
# ============================================================

from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

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
# State is the information carried through the graph.
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
# 4. PERSISTENT MEMORY — SQLITE
#
# Unlike MemorySaver, this checkpointer stores checkpoints
# in a SQLite database file.
#
# Database file:
#
#   memory.db
#
# This file remains after the Python program stops.
# ============================================================

with SqliteSaver.from_conn_string(
    "memory.db"
) as memory:

    # ========================================================
    # SETUP / DEFINITION
    # 5. LLM NODE
    #
    # The LLM receives:
    #
    #   Current message
    #   Previously stored user_name
    #
    # The State is restored by LangGraph before this Node
    # executes.
    # ========================================================

    def llm_node(state: AgentState):

        print("\nLLM NODE: Processing message")

        prompt = f"""
You are a helpful assistant.

Previously remembered user name:
{state["user_name"]}

Current user message:
{state["message"]}

Answer the user's message naturally.

If the user asks for their name and a name is remembered,
tell them their name.
"""

        response = llm.invoke(prompt)

        # ----------------------------------------------------
        # Simple learning example:
        # Detect a name statement and store it in State.
        # ----------------------------------------------------

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


    # ========================================================
    # SETUP / DEFINITION
    # 6. CREATE GRAPH
    # ========================================================

    graph = StateGraph(AgentState)


    # ========================================================
    # SETUP / DEFINITION
    # 7. ADD LLM NODE
    # ========================================================

    graph.add_node(
        "llm_node",
        llm_node
    )


    # ========================================================
    # SETUP / DEFINITION
    # 8. ADD EDGES
    #
    # START
    #   ↓
    # LLM Node
    #   ↓
    # END
    # ========================================================

    graph.add_edge(
        START,
        "llm_node"
    )

    graph.add_edge(
        "llm_node",
        END
    )


    # ========================================================
    # SETUP / DEFINITION
    # 9. COMPILE GRAPH WITH SQLITE CHECKPOINTER
    #
    # This is the key difference from Lesson 30.
    #
    # Lesson 30:
    #   MemorySaver()
    #
    # Lesson 31:
    #   SqliteSaver
    #
    # State is persisted in memory.db.
    # ========================================================

    app = graph.compile(
        checkpointer=memory
    )


    # ========================================================
    # RUNTIME FLOW
    # 10. THREAD ID
    #
    # thread_id identifies the conversation.
    #
    # Using the SAME thread_id allows LangGraph to restore
    # the State from this conversation.
    # ========================================================

    config = {
        "configurable": {
            "thread_id": "user_raj"
        }
    }


    # ========================================================
    # RUNTIME FLOW
    # 11. GET USER MESSAGE
    # ========================================================

    message = input(
        "\nEnter a message: "
    )


    # ========================================================
    # RUNTIME FLOW
    # 12. EXECUTE GRAPH
    #
    # IMPORTANT:
    #
    # If this thread already exists in memory.db,
    # LangGraph restores its previous State.
    #
    # Otherwise, this becomes a new conversation.
    # ========================================================

    result = app.invoke(
    {
        "message": message
    },
    config
)


    # ========================================================
    # RUNTIME FLOW
    # 13. FINAL STATE
    # ========================================================

    print("\nFINAL STATE:")

    print(result)


    # ========================================================
    # RUNTIME FLOW
    # 14. FINAL ANSWER
    # ========================================================

    print("\nFINAL ANSWER:")

    print(result["answer"])


# ============================================================
# END OF PROGRAM
#
# The Python program can now stop.
#
# memory.db remains on disk.
#
# When the program runs again with the same thread_id,
# LangGraph can restore the previous State.
# ============================================================