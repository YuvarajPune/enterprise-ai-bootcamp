# ============================================================
# LESSON 40 — ADVANCED LLM TOOL CALLING
#
# Experiment:
# Fallback Tool
#
# Goal:
# Demonstrate switching to a backup capability when the
# primary tool fails.
#
# Architecture:
#
# User
#   ↓
# LLM
#   ↓
# Primary Tool
#   ↓
# Failure
#   ↓
# Fallback Tool
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
# 3. PRIMARY TOOL
# ============================================================

def get_customer_from_database(customer_id: str):

    raise ConnectionError(
        "Primary customer database is unavailable."
    )


# ============================================================
# 4. FALLBACK TOOL
# ============================================================

def get_customer_from_backup_api(customer_id: str):

    return {
        "customer_id": customer_id,
        "name": "Rahul Sharma",
        "status": "Active",
        "source": "Backup API"
    }


# ============================================================
# 5. TOOL REGISTRY
# ============================================================

TOOL_REGISTRY = {
    "get_customer_from_database":
        get_customer_from_database,

    "get_customer_from_backup_api":
        get_customer_from_backup_api
}


# ============================================================
# 6. PRIMARY TOOL SCHEMA
# ============================================================

primary_tool = {
    "type": "function",
    "name": "get_customer_from_database",
    "description": "Retrieve customer information from the primary database.",
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
# 8. LLM DECIDES PRIMARY TOOL
# ============================================================

interaction = client.interactions.create(
    model="gemini-3.6-flash",
    input=user_request,
    tools=[primary_tool]
)


# ============================================================
# 9. PROCESS TOOL CALL
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
    # 10. PRIMARY TOOL EXECUTION
    # ========================================================

    try:

        result = TOOL_REGISTRY[tool_name](**arguments)

        print("\nPRIMARY TOOL: SUCCESS")
        print("Result:", result)

    except Exception as error:

        print("\nPRIMARY TOOL: FAILED")
        print("Error:", error)


        # ====================================================
        # 11. FALLBACK
        # ====================================================

        print("\nSWITCHING TO FALLBACK TOOL")

        fallback_result = (
            get_customer_from_backup_api(
                **arguments
            )
        )

        print("\nFALLBACK TOOL: SUCCESS")
        print("Result:", fallback_result)