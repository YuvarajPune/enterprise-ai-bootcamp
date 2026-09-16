# -----------------------------
# Import Groq
# -----------------------------

import os
from groq import Groq


# -----------------------------
# Create Groq Client
# -----------------------------

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)


# -----------------------------
# Send Question to Groq
# -----------------------------

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": "Explain AI agents in one sentence."
        }
    ]
)


# -----------------------------
# Display Response
# -----------------------------

print(response.choices[0].message.content)