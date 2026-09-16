# ============================================================
# LLM RESPONSE TIME EXPERIMENT
#
# Goal:
# Measure the end-to-end response time of an LLM API call.
#
# This measures:
# Request sent → Gemini processing → Response received
# ============================================================

from dotenv import load_dotenv
from google import genai
import os
import time


# ============================================================
# 1. LOAD API KEY
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

client = genai.Client(api_key=api_key)


# ============================================================
# 2. LLM REQUEST
# ============================================================

prompt = "Explain artificial intelligence in two sentences."


# ============================================================
# 3. MEASURE RESPONSE TIME
# ============================================================

start_time = time.perf_counter()

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt
)

end_time = time.perf_counter()


# ============================================================
# 4. RESULTS
# ============================================================

response_time = end_time - start_time

print("\n" + "=" * 60)
print("LLM RESPONSE")
print("=" * 60)

print(response.text)

print("\n" + "=" * 60)
print("RESPONSE TIME")
print("=" * 60)

print(f"{response_time:.2f} seconds")