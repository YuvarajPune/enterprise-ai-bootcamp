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
    answer = "Today's weather is 26°C with showers."
else:
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=question
    )
    answer = response.text

print("\nAnswer:\n")
print(answer)