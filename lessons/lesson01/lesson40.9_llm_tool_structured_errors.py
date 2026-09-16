# ============================================================
# LESSON 40 — ADVANCED LLM TOOL CALLING
#
# Experiment 40.9:
# Structured Tool Errors
#
# Goal:
# Demonstrate how a tool can return structured error
# information that the application can interpret.
#
# Architecture:
#
# User
#   ↓
# LLM
#   ↓
# Tool Call
#   ↓
# Tool Execution
#   ↓
# Structured Result
#   ↓
# Application Error Handling
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

    # Simulate a database outage.

    return {
        "success": False,
        "error_code": "DATABASE_UNAVAILABLE",
        "message": "Customer database is temporarily unavailable."
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

customer_tool = {
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
    tools=[customer_tool]
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
    # 9. EXECUTE TOOL
    # ========================================================

    result = TOOL_REGISTRY[tool_name](**arguments)

    print("\n" + "=" * 60)
    print("TOOL RESULT")
    print("=" * 60)

    print(result)


    # ========================================================
    # 10. INTERPRET STRUCTURED RESULT
    # ========================================================

    if result.get("success") is True:

        print("\nRESULT STATUS: SUCCESS")

    else:

        error_code = result.get("error_code")
        message = result.get("message")

        print("\nRESULT STATUS: ERROR")
        print("Error code:", error_code)
        print("Message:", message)


        # ====================================================
        # 11. APPLICATION RECOVERY POLICY
        # ====================================================

        if error_code == "DATABASE_UNAVAILABLE":

            print("\nRECOVERY ACTION: RETRY")

        elif error_code == "RATE_LIMITED":

            print("\nRECOVERY ACTION: WAIT AND RETRY")

        elif error_code == "PERMISSION_DENIED":

            print("\nRECOVERY ACTION: STOP")

        elif error_code == "INVALID_CUSTOMER":

            print("\nRECOVERY ACTION: ASK USER")

        else:

            print("\nRECOVERY ACTION: ESCALATE")