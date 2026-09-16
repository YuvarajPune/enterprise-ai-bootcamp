# ============================================================
# LESSON 40 — ADVANCED LLM TOOL CALLING
#
# Experiment:
# Tool Result Validation
#
# Goal:
# Demonstrate that successful tool execution does not
# necessarily mean a valid business result.
#
# Architecture:
#
# User
#   ↓
# LLM
#   ↓
# Tool Call
#   ↓
# Input Validation
#   ↓
# Tool Execution
#   ↓
# Result Validation
#   ↓
# Valid / Invalid
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
# 3. TOOL IMPLEMENTATION
# ============================================================

def get_customer(customer_id: str):

    # Simulate successful execution but bad business result

    return {
        "customer_id": customer_id,
        "name": None,
        "status": "ERROR"
    }


# ============================================================
# 4. TOOL REGISTRY
# ============================================================

TOOL_REGISTRY = {
    "get_customer": get_customer
}


# ============================================================
# 5. TOOL SCHEMA
# ============================================================

get_customer_tool = {
    "type": "function",
    "name": "get_customer",
    "description": "Retrieve customer information using a customer ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "customer_id": {
                "type": "string"
            }
        },
        "required": ["customer_id"]
    }
}


# ============================================================
# 6. USER REQUEST
# ============================================================

user_request = """
Get the customer information for customer C123.
"""


print("\n" + "=" * 60)
print("USER REQUEST")
print("=" * 60)

print(user_request)


# ============================================================
# 7. LLM TOOL CALL
# ============================================================

interaction = client.interactions.create(
    model="gemini-3.6-flash",
    input=user_request,
    tools=[get_customer_tool]
)


# ============================================================
# 8. PROCESS TOOL CALL
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
    # 9. TOOL EXECUTION
    # ========================================================

    try:

        result = TOOL_REGISTRY[tool_name](**arguments)

        print("\nTOOL EXECUTION: SUCCESS")
        print("Raw result:", result)

    except Exception as error:

        print("\nTOOL EXECUTION: FAILED")
        print("Error:", error)
        continue


    # ========================================================
    # 10. RESULT VALIDATION
    # ========================================================

    print("\nRESULT VALIDATION")

    if (
        result.get("status") != "Active"
        or not result.get("name")
    ):

        print("VALIDATION: FAILED")
        print("Reason: Customer result is incomplete or invalid.")

    else:

        print("VALIDATION: PASSED")
        print("Validated result:", result)