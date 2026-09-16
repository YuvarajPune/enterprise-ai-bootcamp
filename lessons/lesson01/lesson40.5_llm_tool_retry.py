# ============================================================
# LLM TOOL RETRY
#
# Goal:
# Demonstrate application-controlled retry when a valid
# LLM tool call fails during execution.
#
# Architecture:
#
# User
#   ↓
# LLM
#   ↓
# Tool Call
#   ↓
# Validation
#   ↓
# Tool Execution
#   ↓
# Failure
#   ↓
# Retry
#   ↓
# Tool Execution
#   ↓
# Success
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

attempt_count = 0


def get_customer(customer_id: str):
    """
    Simulate a temporary database failure.
    First attempt fails; second attempt succeeds.
    """

    global attempt_count

    attempt_count += 1

    if attempt_count == 1:
        raise ConnectionError(
            "Customer database temporarily unavailable."
        )

    return {
        "customer_id": customer_id,
        "name": "Rahul Sharma",
        "status": "Active"
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
    # 9. VALIDATION
    # ========================================================

    if tool_name not in TOOL_REGISTRY:
        print("\nVALIDATION: FAILED")
        print("Reason: Unknown tool")
        continue


    # ========================================================
    # 10. RETRY LOOP
    # ========================================================

    max_retries = 3

    for attempt in range(1, max_retries + 1):

        print(f"\nTool execution attempt: {attempt}")

        try:

            result = TOOL_REGISTRY[tool_name](**arguments)

            print("\nTOOL EXECUTION: SUCCESS")
            print("Result:", result)

            break

        except Exception as error:

            print("TOOL EXECUTION: FAILED")
            print("Error:", error)

            if attempt == max_retries:
                print("\nMaximum retries reached.")