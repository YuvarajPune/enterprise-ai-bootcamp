# ============================================================
# LESSON 33 — PERSISTENT SEMANTIC MEMORY
#
# Goal:
# Store semantic memory permanently in a SQLite database.
#
# LESSON 32:
#   LLM → identify fact → Python dictionary → RAM
#
# LESSON 33:
#   LLM → identify fact → Application → SQLite DB
#
#
# ARCHITECTURE:
#
# User
#   ↓
# LLM
#   ↓
# Identify useful fact
#   ↓
# Agent Application
#   ↓
# SQLite Database
#   ↓
# Memory persists after program stops
#
#
# IMPORTANT:
# The LLM identifies the memory.
# The APPLICATION controls whether/how it is stored.
# SQLite provides persistent storage.
# ============================================================


# ============================================================
# SETUP / DEFINITION
# 1. IMPORTS
# ============================================================

import sqlite3

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
# 3. SQLITE DATABASE
#
# This creates/opens:
#
#   semantic_memory.db
#
# Unlike a Python dictionary, this information is stored
# on disk and survives after the Python program stops.
# ============================================================

connection = sqlite3.connect(
    "semantic_memory.db"
)


# ============================================================
# SETUP / DEFINITION
# 4. CREATE MEMORY TABLE
#
# The table stores:
#
#   key   → name of the memory
#   value → actual remembered information
# ============================================================

connection.execute(
    """
    CREATE TABLE IF NOT EXISTS semantic_memory (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    """
)

connection.commit()


# ============================================================
# SETUP / DEFINITION
# 5. LLM MEMORY EXTRACTION
#
# The LLM identifies whether the user's message contains
# a useful semantic memory.
#
# For this lesson we are looking specifically for:
#
#   report_format
#
# Example:
#
# "My preferred report format is Excel"
#
# LLM should identify:
#
#   report_format = Excel
# ============================================================

def extract_memory(message):

    prompt = f"""
You are a memory extraction assistant.

User message:
{message}

Determine whether the user has stated their preferred
report format.

If they have, respond exactly in this format:

report_format=VALUE

If they have not, respond:

none

Do not provide any other text.
"""

    response = llm.invoke(prompt)

    return response.content.strip()


# ============================================================
# SETUP / DEFINITION
# 6. APPLICATION MEMORY STORAGE
#
# This is APPLICATION LOGIC.
#
# The application receives the LLM's output and decides
# whether to save it.
#
# The LLM does NOT directly write to the database.
# ============================================================

def save_memory(memory_result):

    if not memory_result.startswith(
        "report_format="
    ):

        return "No memory stored."

    value = memory_result.split(
        "=", 1
    )[1].strip()

    connection.execute(
        """
        INSERT OR REPLACE INTO semantic_memory
        (key, value)
        VALUES (?, ?)
        """,
        ("report_format", value)
    )

    connection.commit()

    return (
        f"Stored semantic memory: "
        f"report_format = {value}"
    )


# ============================================================
# SETUP / DEFINITION
# 7. MEMORY RETRIEVAL
#
# The application retrieves the stored semantic memory
# from SQLite.
# ============================================================

def retrieve_memory():

    cursor = connection.execute(
        """
        SELECT key, value
        FROM semantic_memory
        """
    )

    rows = cursor.fetchall()

    memories = {}

    for key, value in rows:

        memories[key] = value

    return memories


# ============================================================
# RUNTIME FLOW
# 8. CONVERSATION 1
#
# User provides information that may become memory.
# ============================================================

message_1 = input(
    "\nConversation 1 — Enter a message: "
)


# ============================================================
# RUNTIME FLOW
# 9. LLM IDENTIFIES MEMORY
# ============================================================

memory_result = extract_memory(
    message_1
)

print("\nLLM MEMORY EXTRACTION:")

print(memory_result)


# ============================================================
# RUNTIME FLOW
# 10. APPLICATION DECIDES TO STORE
# ============================================================

storage_result = save_memory(
    memory_result
)

print("\nAPPLICATION MEMORY ACTION:")

print(storage_result)


# ============================================================
# RUNTIME FLOW
# 11. SHOW STORED MEMORY
# ============================================================

stored_memory = retrieve_memory()

print("\nPERSISTENT SEMANTIC MEMORY:")

print(stored_memory)


# ============================================================
# RUNTIME FLOW
# 12. CONVERSATION 2
#
# This represents a later interaction.
# ============================================================

message_2 = input(
    "\nConversation 2 — Enter a message: "
)


# ============================================================
# RUNTIME FLOW
# 13. RETRIEVE MEMORY FROM SQLITE
# ============================================================

stored_memory = retrieve_memory()

print("\nRETRIEVED MEMORY:")

print(stored_memory)


# ============================================================
# RUNTIME FLOW
# 14. LLM USES THE MEMORY
#
# The LLM receives:
#
#   Current question
#   Retrieved semantic memory
# ============================================================

prompt = f"""
You are a helpful assistant.

Remembered user information:
{stored_memory}

Current user message:
{message_2}

Use the remembered information when it is relevant.

Give a short, natural answer.
"""

response = llm.invoke(
    prompt
)


# ============================================================
# RUNTIME FLOW
# 15. FINAL STATE
# ============================================================

final_state = {
    "current_message": message_2,
    "semantic_memory": stored_memory,
    "answer": response.content
}

print("\nFINAL STATE:")

print(final_state)


# ============================================================
# RUNTIME FLOW
# 16. FINAL ANSWER
# ============================================================

print("\nFINAL ANSWER:")

print(response.content)


# ============================================================
# RUNTIME FLOW
# 17. CLOSE DATABASE CONNECTION
# ============================================================

connection.close()