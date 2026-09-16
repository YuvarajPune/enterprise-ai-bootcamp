import chromadb

# -----------------------------
# Create Persistent Vector DB
# -----------------------------

client = chromadb.PersistentClient(
    path="vector_db"
)

# -----------------------------
# Create Collection
# -----------------------------

collection = client.get_or_create_collection(
    name="memory"
)

# -----------------------------
# Add Memories
# -----------------------------

memories = [
    "I prefer MacBook",
    "I work with Azure",
    "I prefer concise answers",
    "My son plays basketball"
]

collection.add(
    documents=memories,
    ids=["memory1", "memory2", "memory3", "memory4"]
)

print("Memories stored successfully.")

# -----------------------------
# Search Memory
# -----------------------------

question = "Which computer would I like?"

results = collection.query(
    query_texts=[question],
    n_results=2
)

print("\nSEARCH RESULTS:")
print(results)