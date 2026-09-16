# ============================================================
# LESSON 40 — ADVANCED LLM TOOL CALLING
#
# Goal:
# Handle multiple native tool calls through a tool dispatcher
# and continue the LLM interaction after tool execution.
#
# Previously covered:
# - Basic function calling
# - Tool registry
# - LLM-based tool selection
# - Agent loop
# - Multi-tool agents
# - Tool chaining
#
# This lesson focuses on:
# - Multiple tool definitions
# - Tool dispatcher
# - Tool-call IDs
# - Tool result handling
# - Passing tool results back to the LLM
# - Stateful continuation using previous_interaction_id
#
# ARCHITECTURE:
#
# USER
#   |
#   v
# LLM
#   |
#   | function_call
#   v
# TOOL DISPATCHER
#   |
#   +------> get_temperature()
#   |
#   +------> calculate()
#   |
#   v
# TOOL RESULT
#   |
#   v
# LLM
#   |
#   v
# FINAL ANSWER
#
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

from dotenv import load_dotenv
from google import genai
import os
import json


# ============================================================
# 2. LOAD API KEY
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

client = genai.Client(api_key=api_key)


# ============================================================
# 3. TOOL IMPLEMENTATIONS
# ============================================================

def get_temperature(city: str) -> int:
    """Return the current temperature for a city."""

    temperatures = {
        "Pune": 28,
        "Mumbai": 30,
        "Delhi": 34
    }

    return temperatures.get(city, 25)


def calculate(a: int, b: int) -> int:
    """Add two numbers."""

    return a + b


# ============================================================
# 4. TOOL REGISTRY
#
# Maps the tool name produced by the LLM to the actual
# Python function that executes the capability.
# ============================================================

TOOL_REGISTRY = {
    "get_temperature": get_temperature,
    "calculate": calculate
}


# ============================================================
# 5. TOOL SCHEMAS
#
# These are the contracts exposed to the LLM.
# ============================================================

get_temperature_tool = {
    "type": "function",
    "name": "get_temperature",
    "description": "Get the current temperature for a city.",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "Name of the city"
            }
        },
        "required": ["city"]
    }
}


calculate_tool = {
    "type": "function",
    "name": "calculate",
    "description": "Add two numbers together.",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {
                "type": "integer",
                "description": "First number"
            },
            "b": {
                "type": "integer",
                "description": "Second number"
            }
        },
        "required": ["a", "b"]
    }
}


tools = [
    get_temperature_tool,
    calculate_tool
]


# ============================================================
# 6. TOOL DISPATCHER
#
# The dispatcher receives a native function call from the LLM,
# finds the corresponding implementation, and executes it.
# ============================================================

def dispatch_tool(function_call):

    tool_name = function_call.name
    arguments = dict(function_call.arguments)

    print("\nTOOL REQUEST")
    print("Tool:", tool_name)
    print("Arguments:", arguments)
    print("Call ID:", function_call.id)

    if tool_name not in TOOL_REGISTRY:
        raise ValueError(
            f"Unknown tool requested: {tool_name}"
        )

    tool_function = TOOL_REGISTRY[tool_name]

    result = tool_function(**arguments)

    print("Tool result:", result)

    return result


# ============================================================
# 7. USER REQUEST
# ============================================================

user_request = """
What is the temperature in Pune and Mumbai?
"""

print("\n" + "=" * 60)
print("USER REQUEST")
print("=" * 60)
print(user_request)


# ============================================================
# 8. LLM — FIRST TOOL CALL
# ============================================================

interaction = client.interactions.create(
    model="gemini-3.6-flash",
    input=user_request,
    tools=tools
)


# ============================================================
# 9. EXECUTE TOOL CALLS
# ============================================================

function_results = []

for step in interaction.steps:

    if step.type == "function_call":

        result = dispatch_tool(step)

        function_results.append(
            {
                "type": "function_result",
                "call_id": step.id,
                "name": step.name,
                "result": [
                    {
                        "type": "text",
                        "text": str(result)
                    }
                ]
            }
        )


# ============================================================
# 10. RETURN TOOL RESULTS TO LLM
# ============================================================

interaction = client.interactions.create(
    model="gemini-3.6-flash",
    previous_interaction_id=interaction.id,
    input=function_results,
    tools=tools
)


# ============================================================
# 11. HANDLE ANY ADDITIONAL TOOL CALL
# ============================================================

function_results = []

for step in interaction.steps:

    if step.type == "function_call":

        result = dispatch_tool(step)

        function_results.append(
            {
                "type": "function_result",
                "call_id": step.id,
                "name": step.name,
                "result": [
                    {
                        "type": "text",
                        "text": str(result)
                    }
                ]
            }
        )


# ============================================================
# 12. FINAL RESPONSE
# ============================================================

if function_results:

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        previous_interaction_id=interaction.id,
        input=function_results
    )


print("\n" + "=" * 60)
print("FINAL ANSWER")
print("=" * 60)

print(interaction.output_text)