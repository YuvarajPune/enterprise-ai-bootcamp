# ============================================================
# LLM TOOL FAILURE HANDLING
#
# Goal:
# Handle a failure that occurs during tool execution.
#
# Flow:
#
# User → LLM → Tool Call → Validation → Tool Execution
#                                      ↓
#                                   Failure
#                                      ↓
#                              Error Handling
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
# 2. TOOL IMPLEMENTATION
# ============================================================

def get_customer(customer_id: str):
    """
    Simulate a customer database failure.
    """

    raise ConnectionError(
        "Customer database is unavailable."
    )


# ============================================================
# 3. TOOL REGISTRY
# ============================================================

TOOL_REGISTRY = {
    "get_customer": get_customer
}


# ============================================================
# 4. TOOL SCHEMA
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
# 5. USER REQUEST
# ============================================================

user_request = """
Get the customer information for customer C123.
"""


print("\n" + "=" * 60)
print("USER REQUEST")
print("=" * 60)

print(user_request)


# ============================================================
# 6. LLM TOOL CALL
# ============================================================

interaction = client.interactions.create(
    model="gemini-3.6-flash",
    input=user_request,
    tools=[get_customer_tool]
)


# ============================================================
# 7. VALIDATE AND EXECUTE
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
    # Validation
    # --------------------------------------------------------

    if tool_name not in TOOL_REGISTRY:
        print("\nVALIDATION: FAILED")
        print("Reason: Unknown tool")
        continue


    # --------------------------------------------------------
    # Tool execution
    # --------------------------------------------------------

    try:

        result = TOOL_REGISTRY[tool_name](**arguments)

        print("\nTOOL RESULT:")
        print(result)

    except Exception as error:

        print("\nTOOL EXECUTION: FAILED")
        print("Error:", error)