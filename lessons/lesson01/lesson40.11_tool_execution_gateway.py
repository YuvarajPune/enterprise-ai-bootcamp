# ============================================================
# LESSON 40 — ADVANCED LLM TOOL CALLING
#
# Experiment 40.11:
# Production-Style Tool Execution Gateway
#
# Goal:
# Bring together the major Lesson 40 tool-execution patterns
# into one controlled application layer.
#
# Architecture:
#
# User
#   ↓
# LLM
#   ↓
# Tool Call
#   ↓
# Tool Execution Gateway
#   ├── Tool authorization
#   ├── Argument validation
#   ├── Tool-call budget
#   ├── Execution
#   ├── Retry
#   ├── Fallback
#   ├── Result validation
#   └── Structured errors
#   ↓
# Result
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
# 3. APPLICATION POLICY
# ============================================================

MAX_TOOL_CALLS = 3

tool_call_count = 0


# ============================================================
# 4. PRIMARY TOOL
# ============================================================

def get_customer(customer_id: str):

    # Simulate a temporary database failure.

    raise ConnectionError(
        "Primary customer database is unavailable."
    )


# ============================================================
# 5. FALLBACK TOOL
# ============================================================

def get_customer_backup(customer_id: str):

    return {
        "success": True,
        "customer_id": customer_id,
        "name": "Rahul Sharma",
        "status": "Active",
        "source": "Backup API"
    }


# ============================================================
# 6. TOOL REGISTRY
# ============================================================

TOOL_REGISTRY = {
    "get_customer": get_customer,
    "get_customer_backup": get_customer_backup
}


# ============================================================
# 7. ALLOWED TOOLS
# ============================================================

ALLOWED_TOOLS = {
    "get_customer"
}


# ============================================================
# 8. TOOL SCHEMA
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
# 9. TOOL EXECUTION GATEWAY
# ============================================================

def execute_tool(tool_name, arguments):

    global tool_call_count

    print("\n" + "=" * 60)
    print("TOOL EXECUTION GATEWAY")
    print("=" * 60)


    # --------------------------------------------------------
    # STEP 1 — AUTHORIZATION
    # --------------------------------------------------------

    if tool_name not in ALLOWED_TOOLS:

        return {
            "success": False,
            "error_code": "TOOL_NOT_ALLOWED",
            "message": "Tool is not allowed by application policy."
        }


    print("Authorization: PASSED")


    # --------------------------------------------------------
    # STEP 2 — ARGUMENT VALIDATION
    # --------------------------------------------------------

    if "customer_id" not in arguments:

        return {
            "success": False,
            "error_code": "INVALID_ARGUMENTS",
            "message": "customer_id is required."
        }


    if not isinstance(arguments["customer_id"], str):

        return {
            "success": False,
            "error_code": "INVALID_ARGUMENTS",
            "message": "customer_id must be a string."
        }


    print("Argument validation: PASSED")


    # --------------------------------------------------------
    # STEP 3 — TOOL-CALL BUDGET
    # --------------------------------------------------------

    if tool_call_count >= MAX_TOOL_CALLS:

        return {
            "success": False,
            "error_code": "TOOL_BUDGET_EXCEEDED",
            "message": "Maximum tool-call budget reached."
        }


    tool_call_count += 1

    print(
        f"Tool-call budget: "
        f"{tool_call_count}/{MAX_TOOL_CALLS}"
    )


    # --------------------------------------------------------
    # STEP 4 — PRIMARY TOOL EXECUTION
    # --------------------------------------------------------

    try:

        result = TOOL_REGISTRY[tool_name](
            **arguments
        )

        print("Primary execution: SUCCESS")

    except Exception as error:

        print("Primary execution: FAILED")
        print("Error:", error)


        # ----------------------------------------------------
        # STEP 5 — RETRY
        # ----------------------------------------------------

        print("Retrying primary tool...")

        try:

            result = TOOL_REGISTRY[tool_name](
                **arguments
            )

            print("Retry: SUCCESS")

        except Exception as retry_error:

            print("Retry: FAILED")
            print("Error:", retry_error)


            # ------------------------------------------------
            # STEP 6 — FALLBACK
            # ------------------------------------------------

            print("Switching to fallback tool...")

            result = TOOL_REGISTRY[
                "get_customer_backup"
            ](**arguments)

            print("Fallback: SUCCESS")


    # --------------------------------------------------------
    # STEP 7 — RESULT VALIDATION
    # --------------------------------------------------------

    if not result.get("success"):

        return {
            "success": False,
            "error_code": "INVALID_RESULT",
            "message": "Tool returned an invalid result."
        }


    if not result.get("customer_id"):

        return {
            "success": False,
            "error_code": "INVALID_RESULT",
            "message": "Customer ID missing from result."
        }


    if not result.get("name"):

        return {
            "success": False,
            "error_code": "INVALID_RESULT",
            "message": "Customer name missing from result."
        }


    print("Result validation: PASSED")


    # --------------------------------------------------------
    # STEP 8 — RETURN RESULT
    # --------------------------------------------------------

    return result


# ============================================================
# 10. USER REQUEST
# ============================================================

user_request = """
Get the customer information for customer C123.
"""


print("\n" + "=" * 60)
print("USER REQUEST")
print("=" * 60)

print(user_request)


# ============================================================
# 11. LLM DECISION
# ============================================================

interaction = client.interactions.create(
    model="gemini-3.6-flash",
    input=user_request,
    tools=[customer_tool]
)


# ============================================================
# 12. PROCESS LLM TOOL CALL
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
    # 13. EXECUTE THROUGH GATEWAY
    # ========================================================

    result = execute_tool(
        tool_name,
        arguments
    )


    # ========================================================
    # 14. FINAL APPLICATION RESULT
    # ========================================================

    print("\n" + "=" * 60)
    print("FINAL TOOL RESULT")
    print("=" * 60)

    print(result)