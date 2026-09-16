import os
import chromadb
from groq import Groq

# -----------------------------
# Groq Client
# -----------------------------

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# -----------------------------
# Vector Database
# -----------------------------

chroma_client = chromadb.PersistentClient(
    path="rag_db"
)

collection = chroma_client.get_or_create_collection(
    name="company_policy"
)


# -----------------------------
# Knowledge Base
# -----------------------------

documents = [
    "Employees can work remotely up to 3 days per week.",
    "Employees receive 24 paid leave days per year.",
    "Business travel requires manager approval."
]

collection.add(
    documents=documents,
    ids=["policy1", "policy2", "policy3"]
)


# -----------------------------
# User Question
# -----------------------------

question = input("Ask a question about company policy: ")


# -----------------------------
# Retrieve Relevant Information
# -----------------------------

results = collection.query(
    query_texts=[question],
    n_results=3
)

# -----------------------------
# Relevance Check
# -----------------------------

distance = results["distances"][0][0]
THRESHOLD = 0.8

if distance > THRESHOLD:

    print("\nNO RELEVANT INFORMATION FOUND.")
    print("The answer is not available in the company policy.")

else:

    retrieved_documents = results["documents"][0]

    print("\nRETRIEVED INFORMATION:")
    print(retrieved_documents)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the user's question using ONLY the provided company policy information. "
                    "Do not use outside knowledge or make assumptions. "
                    "Clearly distinguish between what the policy explicitly states "
                    "and what it does not state. "
                    "If the policy does not provide enough information to answer, "
                    "say: 'The provided policy does not specify this.'"
                )
            },
            {
                "role": "user",
                "content": f"""
Question:
{question}

Company policy information:
{retrieved_documents}
"""
            }
        ]
    )

    answer = response.choices[0].message.content

    print("\nFINAL ANSWER:")
    print(answer)