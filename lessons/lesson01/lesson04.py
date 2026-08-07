import os

from dotenv import load_dotenv
from google import genai


def gemini_choose_tool(question):

    prompt = f"""
You are an AI agent.

Available tools:
1. weather - For weather questions
2. calculator - For math calculations

Return ONLY one word.

Examples:
weather
calculator

Question:
{question}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text.strip().lower()

def get_weather(question):
    question = question.lower()

    if "tomorrow" in question:
        return "Tomorrow will be 28°C with light rain."

    elif "today" in question:
        return "Today is 26°C with showers."

    else:
        return "Please specify today or tomorrow."

def calculator(expression):
    return eval(expression)

def choose_tool(question):
    question = question.lower()

    if "weather" in question:
        return "weather"

    if any(op in question for op in ["+", "-", "*", "/"]):
        return "calculator"

    return None

tools = {
    "weather": get_weather,
    "calculator": calculator
}

# Load variables from .env
load_dotenv()

# Read API key
api_key = os.getenv("GEMINI_API_KEY")

print(api_key[:8] + "..." if api_key else "No Key Found")
print(len(api_key) if api_key else 0)

# Create Gemini client
client = genai.Client(api_key=api_key)
print(tools)

question = input("Ask me something: ")

tool = gemini_choose_tool(question)
print("Gemini chose:", tool)

if tool in tools:
    answer = tools[tool](question)
else:
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=question
    )
    answer = response.text

print("\nAnswer:\n")
print(answer)