# ============================================================
# LESSON 36 — SEMANTIC MEMORY WITH VECTOR DB
#
# Goal:
# Understand semantic retrieval using:
#
#   Embedding Model
#   Vector DB
#   Similarity Search
#   LLM
#
#
# RUNTIME ARCHITECTURE:
#
# Stored Memory
#       ↓
# Embedding Model
#       ↓
# Vector
#       ↓
# ChromaDB
#
#
# User Question
#       ↓
# Embedding Model
#       ↓
# Query Vector
#       ↓
# ChromaDB
#       ↓
# Similarity Search
#       ↓
# Relevant Memory
#       ↓
# LLM
#       ↓
# Final Answer
#
#
# IMPORTANT:
#
# Embedding Model:
#     Text → Vector
#
# Vector DB:
#     Vector → Similarity Search
#
# LLM:
#     Retrieved Memory → Answer
# ============================================================


# ============================================================
# SETUP / DEFINITION
# 1. IMPORTS
# ============================================================

import os
import chromadb

from google import genai
from google.genai import types

from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()

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
# 3. GOOGLE GENAI CLIENT
#
# The embedding model will be called through the Google
# GenAI API.
#
# The API key should already be available in the environment
# as GEMINI_API_KEY.
# ============================================================

client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY")
)


# ============================================================
# SETUP / DEFINITION
# 4. EMBEDDING MODEL
#
# This model converts text into a vector.
#
# Text
#   ↓
# Gemini Embedding Model
#   ↓
# Vector
#
# We use gemini-embedding-001 for this lesson.
# ============================================================

EMBEDDING_MODEL = "gemini-embedding-001"


# ============================================================
# SETUP / DEFINITION
# 5. CHROMA VECTOR DB
#
# PersistentClient stores the Vector DB on disk.
#
# Directory:
#
#     chroma_memory/
#
# The database therefore survives after the Python program
# stops.
# ============================================================

chroma_client = chromadb.PersistentClient(
    path="chroma_memory"
)


# ============================================================
# SETUP / DEFINITION
# 6. MEMORY COLLECTION
#
# A collection contains:
#
#   IDs
#   Original documents
#   Embeddings
# ============================================================

collection = chroma_client.get_or_create_collection(
    name="semantic_memory"
)


# ============================================================
# SETUP / DEFINITION
# 7. EMBEDDING FUNCTION
#
# This function sends text to the embedding model and
# returns the numerical vector.
# ============================================================

def create_embedding(text):

    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT"
        )
    )

    return result.embeddings[0].values


# ============================================================
# SETUP / DEFINITION
# 8. MEMORIES
#
# These are semantic facts/preferences about the user.
# ============================================================

memories = [
    "My preferred report format is Excel.",
    "I prefer concise executive summaries.",
    "I usually work with AI and data transformation projects.",
    "I prefer meetings in the morning."
]


# ============================================================
# SETUP / DEFINITION
# 9. CREATE EMBEDDINGS FOR MEMORIES
#
# Memory text
#      ↓
# Embedding Model
#      ↓
# Vector
# ============================================================

memory_embeddings = []

for memory in memories:

    embedding = create_embedding(
        memory
    )

    memory_embeddings.append(
        embedding
    )


# ============================================================
# SETUP / DEFINITION
# 10. STORE MEMORIES IN VECTOR DB
#
# Chroma stores:
#
#   ID
#   Original text
#   Embedding vector
# ============================================================

collection.upsert(
    ids=[
        "memory_1",
        "memory_2",
        "memory_3",
        "memory_4"
    ],
    documents=memories,
    embeddings=memory_embeddings
)


# ============================================================
# RUNTIME FLOW
# 11. GET USER QUESTION
# ============================================================

question = input(
    "\nAsk a question: "
)


# ============================================================
# RUNTIME FLOW
# 12. CREATE QUERY EMBEDDING
#
# IMPORTANT:
#
# We use the SAME embedding model for:
#
#   Stored memories
#   User question
#
# This puts them into the same vector space.
# ============================================================

query_embedding = client.models.embed_content(
    model=EMBEDDING_MODEL,
    contents=question,
    config=types.EmbedContentConfig(
        task_type="RETRIEVAL_QUERY"
    )
).embeddings[0].values


# ============================================================
# RUNTIME FLOW
# 13. VECTOR DB SEARCH
#
# Chroma compares the query vector with stored vectors.
#
# It returns the most similar memories.
# ============================================================

search_result = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)


# ============================================================
# RUNTIME FLOW
# 14. RETRIEVE MEMORIES
# ============================================================

retrieved_memories = (
    search_result["documents"][0]
)

distances = (
    search_result["distances"][0]
)


# ============================================================
# RUNTIME FLOW
# 15. DISPLAY RETRIEVED MEMORIES
#
# This is important for learning.
#
# We want to SEE what the Vector DB selected.
# ============================================================

print("\nRETRIEVED MEMORIES:")

for memory, distance in zip(
    retrieved_memories,
    distances
):

    print(
        f"Memory: {memory}"
    )

    print(
        f"Distance: {distance}"
    )

    print()


# ============================================================
# RUNTIME FLOW
# 16. PREPARE MEMORY FOR LLM
# ============================================================

memory_text = "\n".join(
    retrieved_memories
)


# ============================================================
# RUNTIME FLOW
# 17. LLM USES RETRIEVED MEMORY
#
# The LLM receives:
#
#   User question
#   Relevant memories retrieved by Vector DB
#
# It does NOT perform the Vector DB search itself.
# ============================================================

prompt = f"""
You are a helpful assistant.

User question:
{question}

Relevant memories retrieved from the user's memory store:

{memory_text}

Use the relevant memory to answer the question.

Do not invent information.

Give a short, natural answer.
"""

response = llm.invoke(
    prompt
)


# ============================================================
# RUNTIME FLOW
# 18. FINAL STATE
# ============================================================

final_state = {

    "question": question,

    "embedding_model": EMBEDDING_MODEL,

    "retrieved_memories": retrieved_memories,

    "distances": distances,

    "answer": response.content
}


print("\nFINAL STATE:")

print(final_state)


# ============================================================
# RUNTIME FLOW
# 19. FINAL ANSWER
# ============================================================

print("\nFINAL ANSWER:")

print(response.content)