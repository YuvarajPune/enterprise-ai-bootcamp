def get_weather(question):
    question = question.lower()

    if "tomorrow" in question:
        return "Tomorrow will be 28°C with light rain."

    elif "today" in question:
        return "Today is 26°C with showers."

    else:
        return "Please specify today or tomorrow."

import os

from dotenv import load_dotenv
from google import genai

# Load variables from .env
load_dotenv()

# Read API key
api_key = os.getenv("GEMINI_API_KEY")

print(api_key[:8] + "..." if api_key else "No Key Found")
print(len(api_key) if api_key else 0)

# Create Gemini client
client = genai.Client(api_key=api_key)
question = input("Ask me something: ")


if "weather" in question.lower():
    answer = get_weather(question)
else:
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=question
    )
    answer = response.text

print("\nAnswer:\n")
print(answer)