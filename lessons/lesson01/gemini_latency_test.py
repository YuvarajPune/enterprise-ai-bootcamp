import os
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

print("Starting Gemini request...")

start = time.perf_counter()

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="Say hello in one word."
)

elapsed = time.perf_counter() - start

print("\nResponse:", response.text)
print(f"Time: {elapsed:.2f} seconds")