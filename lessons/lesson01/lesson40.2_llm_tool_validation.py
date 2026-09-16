# ============================================================
# LLM TOOL VALIDATION
#
# Goal:
# Let the LLM generate a tool call, then validate the
# generated tool name and arguments before execution.
#
# Architecture:
#
# User
#   ↓
# LLM
#   ↓
# Tool Call
#   ↓
# Validation Layer
#   ↓
# Valid? ── No ──> Reject
#   │
#  Yes
#   ↓
# Execute Tool
#   ↓
# Tool Result
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

from dotenv import load_dotenv
from google import genai
import os


# ============================================================
# 2. LOAD API KEY
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found")

client = genai.Client(api_key=api_key)


# ============================================================
# 3. TOOL IMPLEMENTATION
# ============================================================

def calculate(a: int, b: int) -> int:
    return a + b


# ============================================================
# 4. TOOL REGISTRY
# ============================================================

TOOL_REGISTRY = {
    "calculate": calculate
}


# ============================================================
# 5. TOOL SCHEMA
# ============================================================

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


# ============================================================
# 6. USER REQUEST
# ============================================================

user_request = "Calculate 25 + 17."


print("\n" + "=" * 60)
print("USER REQUEST")
print("=" * 60)

print(user_request)


# ============================================================
# 7. LLM GENERATES TOOL CALL
# ============================================================

interaction = client.interactions.create(
    model="gemini-3.6-flash",
    input=user_request,
    tools=[calculate_tool]
)


# ============================================================
# 8. VALIDATE AND EXECUTE TOOL CALL
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


    # --------------------------------------------------------
    # Validate tool name
    # --------------------------------------------------------

    if tool_name not in TOOL_REGISTRY:

        print("\nVALIDATION: FAILED")
        print("Reason: Unknown tool")

        continue


    # --------------------------------------------------------
    # Validate required arguments
    # --------------------------------------------------------

    if "a" not in arguments or "b" not in arguments:

        print("\nVALIDATION: FAILED")
        print("Reason: Missing required arguments")

        continue


    # --------------------------------------------------------
    # Validate argument types
    # --------------------------------------------------------

    if not isinstance(arguments["a"], int):
        print("\nVALIDATION: FAILED")
        print("Reason: 'a' must be an integer")
        continue

    if not isinstance(arguments["b"], int):
        print("\nVALIDATION: FAILED")
        print("Reason: 'b' must be an integer")
        continue


    # --------------------------------------------------------
    # Validation successful
    # --------------------------------------------------------

    print("\nVALIDATION: PASSED")


    # --------------------------------------------------------
    # Execute tool
    # --------------------------------------------------------

    result = TOOL_REGISTRY[tool_name](**arguments)

    print("TOOL RESULT:", result)