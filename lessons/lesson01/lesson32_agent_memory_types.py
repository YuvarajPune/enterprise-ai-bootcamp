# ============================================================
# LESSON 32 — AGENT MEMORY TYPES
#
# Goal:
# Understand SEMANTIC MEMORY.
#
# Semantic Memory:
#   Stores facts, preferences, or information about a user
#   that can be useful in future interactions.
#
# Example:
#
# User:
#   "My preferred report format is Excel."
#
# Memory:
#   report_format = Excel
#
# Later:
#   Agent retrieves the preference
#   and uses Excel as the report format.
#
#
# IMPORTANT:
# This lesson demonstrates the CONCEPT of semantic memory.
# It is not yet using a Vector DB for semantic memory.
#
# RUNTIME ARCHITECTURE:
#
# User
#   ↓
# LLM
#   ↓
# Extract useful fact
#   ↓
# Semantic Memory Store
#   ↓
# Later interaction
#   ↓
# Retrieve fact
#   ↓
# LLM
#   ↓
# Answer
# ============================================================


# ============================================================
# SETUP / DEFINITION
# 1. IMPORTS
# ============================================================

from typing import TypedDict

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
# 3. SEMANTIC MEMORY
#
# For this lesson, we use a simple Python dictionary.
#
# In a production system this could be stored in:
#
#   SQL database
#   NoSQL database
#   Vector database
#   Dedicated memory service
# ============================================================

semantic_memory = {}


# ============================================================
# SETUP / DEFINITION
# 4. STATE
#
# State represents the current interaction.
# ============================================================

class AgentState(TypedDict):

    message: str
    memory: str
    answer: str


# ============================================================
# SETUP / DEFINITION
# 5. MEMORY EXTRACTION
#
# This function identifies a preference from the user's
# message and stores it in semantic memory.
#
# This is a simple learning example.
# ============================================================

def extract_memory(message: str):

    message_lower = message.lower()

    if "preferred report format is" in message_lower:

        preference = (
            message_lower
            .split("preferred report format is", 1)[1]
            .strip()
        )

        semantic_memory["report_format"] = (
            preference.capitalize()
        )

        return (
            f"Stored preference: "
            f"{semantic_memory['report_format']}"
        )

    return "No new semantic memory identified."


# ============================================================
# SETUP / DEFINITION
# 6. MEMORY RETRIEVAL
#
# This function retrieves an existing fact from memory.
# ============================================================

def retrieve_memory():

    if "report_format" in semantic_memory:

        return (
            "User's preferred report format is "
            + semantic_memory["report_format"]
        )

    return "No report format preference is stored."


# ============================================================
# RUNTIME FLOW
# 7. CONVERSATION 1 — STORE MEMORY
# ============================================================

message_1 = input(
    "\nConversation 1 — Enter a message: "
)


# ============================================================
# RUNTIME FLOW
# 8. EXTRACT AND STORE MEMORY
# ============================================================

memory_result = extract_memory(
    message_1
)

print("\nMEMORY ACTION:")

print(memory_result)


# ============================================================
# RUNTIME FLOW
# 9. SHOW MEMORY
# ============================================================

print("\nSEMANTIC MEMORY:")

print(semantic_memory)


# ============================================================
# RUNTIME FLOW
# 10. CONVERSATION 2
#
# This represents a later interaction.
# ============================================================

message_2 = input(
    "\nConversation 2 — Enter a message: "
)


# ============================================================
# RUNTIME FLOW
# 11. RETRIEVE MEMORY
# ============================================================

retrieved_memory = retrieve_memory()

print("\nRETRIEVED MEMORY:")

print(retrieved_memory)


# ============================================================
# RUNTIME FLOW
# 12. LLM USES MEMORY
#
# The LLM receives the current question AND the remembered
# fact.
# ============================================================

prompt = f"""
You are a helpful assistant.

Remembered user information:
{retrieved_memory}

Current user message:
{message_2}

Answer the user's message using the remembered information
when it is relevant.
"""

response = llm.invoke(prompt)


# ============================================================
# RUNTIME FLOW
# 13. FINAL STATE
# ============================================================

final_state = {
    "message": message_2,
    "memory": retrieved_memory,
    "answer": response.content
}

print("\nFINAL STATE:")

print(final_state)


# ============================================================
# RUNTIME FLOW
# 14. FINAL ANSWER
# ============================================================

print("\nFINAL ANSWER:")

print(response.content)