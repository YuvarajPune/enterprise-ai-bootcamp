# ============================================================
# LLM TOOL VALIDATION — INVALID TOOL CALL
#
# Goal:
# Demonstrate that an application should reject an invalid
# LLM-generated tool request before execution.
# ============================================================

from dotenv import load_dotenv
from google import genai
import os


# ============================================================
# 1. SETUP
# ============================================================

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# ============================================================
# 2. TOOL
# ============================================================

def calculate(a: int, b: int) -> int:
    return a + b


TOOL_REGISTRY = {
    "calculate": calculate
}


# ============================================================
# 3. TOOL SCHEMA
# ============================================================

calculate_tool = {
    "type": "function",
    "name": "calculate",
    "description": "Add two integers together.",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"}
        },
        "required": ["a", "b"]
    }
}


# ============================================================
# 4. USER REQUEST
# ============================================================

user_request = """
Calculate 25 + 17.

For this experiment, deliberately provide the value
'twenty five' as argument a instead of the number 25.
"""


print("\n" + "=" * 60)
print("USER REQUEST")
print("=" * 60)

print(user_request)


# ============================================================
# 5. LLM
# ============================================================

interaction = client.interactions.create(
    model="gemini-3.6-flash",
    input=user_request,
    tools=[calculate_tool]
)


# ============================================================
# 6. VALIDATION
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

    if tool_name not in TOOL_REGISTRY:
        print("\nVALIDATION: FAILED")
        print("Reason: Unknown tool")
        continue

    if "a" not in arguments or "b" not in arguments:
        print("\nVALIDATION: FAILED")
        print("Reason: Missing required argument")
        continue

    if not isinstance(arguments["a"], int):
        print("\nVALIDATION: FAILED")
        print("Reason: 'a' must be an integer")
        continue

    if not isinstance(arguments["b"], int):
        print("\nVALIDATION: FAILED")
        print("Reason: 'b' must be an integer")
        continue

    print("\nVALIDATION: PASSED")

    result = TOOL_REGISTRY[tool_name](**arguments)

    print("TOOL RESULT:", result)