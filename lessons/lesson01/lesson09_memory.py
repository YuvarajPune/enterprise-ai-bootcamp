import json
import os
from groq import Groq

# -----------------------------
# Memory File
# -----------------------------

MEMORY_FILE = "memory.json"

# -----------------------------
# Groq Client
# -----------------------------

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# -----------------------------
# Load Memory
# -----------------------------

if os.path.exists(MEMORY_FILE):

    with open(MEMORY_FILE, "r") as file:
        memory = json.load(file)

else:

    memory = []


# -----------------------------
# Choose Memory Operation
# -----------------------------

choice = input(
    "Choose an operation:\n"
    "1. Store memory\n"
    "2. Retrieve memory\n"
    "Enter 1 or 2: "
)


# -----------------------------
# Store Memory
# -----------------------------

if choice == "1":

    user_message = input("\nWhat should I remember? ")

    memory.append({
        "type": "preference",
        "topic": "laptop",
        "value": user_message
    })

    with open(MEMORY_FILE, "w") as file:
        json.dump(memory, file, indent=4)

    print("\nMemory stored.")
    print(memory)


# -----------------------------
# Retrieve Memory
# -----------------------------

elif choice == "2":

    memory_question = input("\nAsk about your memory: ")

    # -----------------------------
    # Ask LLM to Identify Topic
    # -----------------------------

    topic_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "Identify the topic of the user's memory question. "
                    "Return only one word. "
                    "For questions about computers or laptops, return: laptop"
                )
            },
            {
                "role": "user",
                "content": memory_question
            }
        ]
    )

    topic = topic_response.choices[0].message.content.strip().lower()

    print("\nIDENTIFIED TOPIC:")
    print(topic)

    # -----------------------------
    # Retrieve Relevant Memory
    # -----------------------------

    retrieved_memory = []

    for item in memory:

        if item.get("topic") == topic:
            retrieved_memory.append(item)

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
                "Answer the user's question using the provided memory. "
                "If the memory does not contain the answer, say you do not know."
            )
        },
        {
            "role": "user",
            "content": f"""
User question:
{memory_question}

Relevant memory:
{retrieved_memory}
"""
        }
    ]
)

answer = answer_response.choices[0].message.content

print("\nFINAL ANSWER:")
print(answer)

