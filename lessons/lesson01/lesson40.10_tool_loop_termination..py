# ============================================================
# LESSON 40 — ADVANCED LLM TOOL CALLING
#
# Experiment 40.10:
# Tool Loop Termination
#
# Goal:
# Demonstrate explicit application-controlled termination
# of an agent's tool-calling loop.
#
# Architecture:
#
# User
#   ↓
# LLM
#   ↓
# Tool Call
#   ↓
# Application executes tool
#   ↓
# Tool result
#   ↓
# LLM
#   ↓
# Final answer OR another tool call
#   ↓
# Application termination policy
#
# Termination conditions:
#   1. Final answer
#   2. Maximum tool calls
#   3. Tool failure
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
# 6. APPLICATION POLICY
# ============================================================

MAX_TOOL_CALLS = 3

tool_call_count = 0


# ============================================================
# 7. USER REQUEST
# ============================================================

user_request = """
Get the customer information for customer C123.
"""


print("\n" + "=" * 60)
print("USER REQUEST")
print("=" * 60)

print(user_request)


# ============================================================
# 8. FIRST LLM CALL
# ============================================================

interaction = client.interactions.create(
    model="gemini-3.6-flash",
    input=user_request,
    tools=[customer_tool]
)


# ============================================================
# 9. PROCESS LLM RESPONSE
# ============================================================

for step in interaction.steps:

    # --------------------------------------------------------
    # FINAL ANSWER
    # --------------------------------------------------------

    if step.type == "text":

        print("\n" + "=" * 60)
        print("FINAL ANSWER")
        print("=" * 60)

        print(step.text)

        break


    # --------------------------------------------------------
    # TOOL CALL
    # --------------------------------------------------------

    if step.type == "function_call":

        # Check application tool budget

        if tool_call_count >= MAX_TOOL_CALLS:

            print("\n" + "=" * 60)
            print("AGENT TERMINATED")
            print("=" * 60)

            print("Reason: Maximum tool-call budget reached.")

            break


        tool_call_count += 1

        tool_name = step.name
        arguments = dict(step.arguments)

        print("\n" + "=" * 60)
        print(f"TOOL CALL #{tool_call_count}")
        print("=" * 60)

        print("Tool:", tool_name)
        print("Arguments:", arguments)


        # ----------------------------------------------------
        # TOOL VALIDATION
        # ----------------------------------------------------

        if tool_name not in TOOL_REGISTRY:

            print("\nAGENT TERMINATED")
            print("Reason: Unknown tool.")

            break


        # ----------------------------------------------------
        # TOOL EXECUTION
        # ----------------------------------------------------

        try:

            result = TOOL_REGISTRY[tool_name](**arguments)

            print("Tool result:", result)

        except Exception as error:

            print("\n" + "=" * 60)
            print("AGENT TERMINATED")
            print("=" * 60)

            print("Reason: Tool execution failure.")
            print("Error:", error)

            break


# ============================================================
# END
# ============================================================