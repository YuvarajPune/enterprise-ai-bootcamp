import os
import chromadb
from groq import Groq

# -----------------------------
# Groq Client
# -----------------------------

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# -----------------------------
# Connect to Vector Database
# -----------------------------

chroma_client = chromadb.PersistentClient(
    path="vector_db"
)

collection = chroma_client.get_or_create_collection(
    name="memory"
)


# -----------------------------
# User Question
# -----------------------------

question = input("Ask about your memory: ")


# -----------------------------
# Semantic Search
# -----------------------------

results = collection.query(
    query_texts=[question],
    n_results=1
)

retrieved_memory = results["documents"][0]


print("\nRETRIEVED MEMORY:")
print(retrieved_memory)


# -----------------------------
# Generate Final Answer
# -----------------------------

answer_response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": (
                "Answer the user's question using the retrieved memory. "
                "If the memory does not contain the answer, say you do not know."
            )
        },
        {
            "role": "user",
            "content": f"""
User question:
{question}

Retrieved memory:
{retrieved_memory}
"""
        }
    ]
)

answer = answer_response.choices[0].message.content

print("\nFINAL ANSWER:")
print(answer)