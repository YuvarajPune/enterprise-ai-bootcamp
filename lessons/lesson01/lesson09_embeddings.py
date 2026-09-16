from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load Embedding Model
# -----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# Memory and User Question
# -----------------------------

memory = "I prefer MacBook"

question = "Which computer would I like?"


# -----------------------------
# Create Embeddings
# -----------------------------

memory_embedding = model.encode([memory])
question_embedding = model.encode([question])


# -----------------------------
# Calculate Similarity
# -----------------------------

similarity = cosine_similarity(
    question_embedding,
    memory_embedding
)[0][0]


# -----------------------------
# Display Results
# -----------------------------

print("Memory:")
print(memory)

print("\nQuestion:")
print(question)

print("\nSimilarity:")
print(similarity)