from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load Embedding Model
# -----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# Memory
# -----------------------------

memories = [
    "I prefer MacBook",
    "I work with Azure",
    "I prefer concise answers",
    "My son plays basketball"
]


# -----------------------------
# User Question
# -----------------------------

question = "Which computer would I like?"


# -----------------------------
# Create Embeddings
# -----------------------------

memory_embeddings = model.encode(memories)
question_embedding = model.encode([question])


# -----------------------------
# Calculate Similarity
# -----------------------------

similarities = cosine_similarity(
    question_embedding,
    memory_embeddings
)[0]


# -----------------------------
# Display Similarities
# -----------------------------

print("QUESTION:")
print(question)

print("\nSIMILARITIES:")

for memory, score in zip(memories, similarities):

    print(f"{score:.4f}  →  {memory}")