# ============================================================
# LESSON 40 — ADVANCED LLM TOOL CALLING
#
# Experiment:
# Controlling Tool Use
#
# Goal:
# Demonstrate how the application can restrict which tools
# are available to the LLM.
#
# Architecture:
#
# User
#   ↓
# Application Policy
#   ↓
# Allowed Tools
#   ↓
# LLM
#   ↓
# Tool Call
#   ↓
# Application
#   ↓
# Tool Execution
#
# IMPORTANT:
# The LLM can only choose from tools supplied to it.
# The application controls the available capabilities.
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

from dotenv import load_dotenv
from google import genai
import os


# ============================================================
# 2. SETUP
# ============================================================

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# ============================================================
# 3. TOOL IMPLEMENTATIONS
# ============================================================

def get_temperature(city: str):

    temperatures = {
        "Pune": 28,
        "Mumbai": 30
    }

    return temperatures.get(city, 25)


def calculate(a: int, b: int):

    return a + b


# ============================================================
# 4. TOOL REGISTRY
# ============================================================

TOOL_REGISTRY = {
    "get_temperature": get_temperature,
    "calculate": calculate
}


# ============================================================
# 5. ALL AVAILABLE TOOL SCHEMAS
# ============================================================

weather_tool = {
    "type": "function",
    "name": "get_temperature",
    "description": "Get the current temperature of a city.",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string"
            }
        },
        "required": ["city"]
    }
}


calculator_tool = {
    "type": "function",
    "name": "calculate",
    "description": "Add two integers.",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {
                "type": "integer"
            },
            "b": {
                "type": "integer"
            }
        },
        "required": ["a", "b"]
    }
}


# ============================================================
# 6. APPLICATION TOOL POLICY
# ============================================================

# The application decides that only the calculator
# is allowed for this request.

ALLOWED_TOOLS = [
    calculator_tool
]


# ============================================================
# 7. USER REQUEST
# ============================================================

user_request = """
What is 25 plus 17?
"""


print("\n" + "=" * 60)
print("USER REQUEST")
print("=" * 60)

print(user_request)


# ============================================================
# 8. SHOW APPLICATION POLICY
# ============================================================

print("\n" + "=" * 60)
print("APPLICATION TOOL POLICY")
print("=" * 60)

print("Allowed tools:")

for tool in ALLOWED_TOOLS:
    print("-", tool["name"])


# ============================================================
# 9. LLM
# ============================================================

interaction = client.interactions.create(
    model="gemini-3.6-flash",
    input=user_request,
    tools=ALLOWED_TOOLS
)


# ============================================================
# 10. PROCESS LLM TOOL CALL
# ============================================================

for step in interaction.steps:

    if step.type != "function_call":
        continue

    tool_name = step.name
    arguments = dict(step.arguments)

    print("\n" + "=" * 60)
    print("LLM TOOL CALL")
    print("=" * 60)

    print("Tool:", tool_name)
    print("Arguments:", arguments)


    # ========================================================
    # 11. APPLICATION VALIDATION
    # ========================================================

    allowed_tool_names = {
        tool["name"]
        for tool in ALLOWED_TOOLS
    }

    if tool_name not in allowed_tool_names:

        print("\nTOOL BLOCKED")
        print("Reason: Tool is not allowed by application policy.")

        continue


    # ========================================================
    # 12. TOOL EXECUTION
    # ========================================================

    result = TOOL_REGISTRY[tool_name](**arguments)

    print("\nTOOL EXECUTION: SUCCESS")
    print("Result:", result)