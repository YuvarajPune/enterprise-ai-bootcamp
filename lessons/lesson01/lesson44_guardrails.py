"""
========================================
LESSON 44 — GUARDRAILS
========================================

Goal:
Understand how application-controlled guardrails
enforce enterprise policies around agent actions.

Architecture:

User Request
     ↓
    LLM
     ↓
Structured Operation
     ↓
Validation
     ↓
Guardrails
     ├── ALLOW → Execute
     ├── HITL  → Human approval
     └── BLOCK → Stop
     ↓
Tool Execution

Key principle:
The LLM proposes the action.
The application validates and enforces policy.
The tool executes only after policy checks pass.
"""

import os
import re

from dotenv import load_dotenv
from google import genai


# ========================================
# SETUP
# ========================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)


# ========================================
# POLICY CONFIGURATION
# ========================================

ALLOWED_ACTIONS = {
    "view_customer",
    "view_order",
    "refund"
}

REFUND_AUTO_LIMIT = 10000


# ========================================
# LLM — INTERPRET USER REQUEST
# ========================================

def interpret_request(user_request):

    prompt = f"""
You are an enterprise AI assistant.

Interpret the user's request.

Return exactly one line in this format:

ACTION=<action>;CUSTOMER=<customer_id>;AMOUNT=<amount>

Allowed actions:
- view_customer
- view_order
- refund

If the request does not contain an amount, use 0.

Do not explain your answer.
Do not add markdown.
Do not add any other text.

User request:
{user_request}
"""

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt
    )

    response = interaction.output_text.strip()

    print("\nLLM interpretation:")
    print(response)

    return response


# ========================================
# PARSE AND VALIDATE LLM OUTPUT
# ========================================

def parse_operation(response):

    pattern = (
        r"ACTION=(.*?);"
        r"CUSTOMER=(.*?);"
        r"AMOUNT=(.*)"
    )

    match = re.search(pattern, response)

    if not match:
        raise ValueError(
            "LLM did not return the required structured operation format"
        )

    action = match.group(1).strip().lower()
    customer = match.group(2).strip()

    try:
        amount = float(match.group(3).strip())
    except ValueError:
        raise ValueError("Amount returned by LLM is not numeric")

    if not action:
        raise ValueError("Action is empty")

    if not customer:
        raise ValueError("Customer is empty")

    if amount < 0:
        raise ValueError("Amount cannot be negative")

    operation = {
        "action": action,
        "customer": customer,
        "amount": amount
    }

    print("\nValidated operation:")
    print(operation)

    return operation


# ========================================
# GUARDRAIL
# ========================================

def check_guardrail(operation):

    action = operation["action"]
    amount = operation["amount"]

    # ------------------------------------
    # Guardrail 1 — Action-level policy
    # ------------------------------------

    if action not in ALLOWED_ACTIONS:

        return {
            "decision": "BLOCK",
            "reason": f"Action '{action}' is prohibited"
        }

    # ------------------------------------
    # Guardrail 2 — Parameter-level policy
    # ------------------------------------

    if action == "refund" and amount > REFUND_AUTO_LIMIT:

        return {
            "decision": "HITL",
            "reason": (
                f"Refund exceeds automatic limit of "
                f"₹{REFUND_AUTO_LIMIT:,.2f}"
            )
        }

    # ------------------------------------
    # Policy passed
    # ------------------------------------

    return {
        "decision": "ALLOW",
        "reason": "Action complies with policy"
    }


# ========================================
# HUMAN APPROVAL
# ========================================

def request_human_approval(operation):

    print("\n========================================")
    print("HUMAN APPROVAL REQUIRED")
    print("========================================")

    print(f"Action   : {operation['action']}")
    print(f"Customer : {operation['customer']}")
    print(f"Amount   : ₹{operation['amount']:,.2f}")

    print("\nThis action requires human authorization.")

    decision = input(
        "\nApprove this action? (yes/no): "
    ).strip().lower()

    if decision == "yes":

        print("\nHuman decision: APPROVED")
        return True

    print("\nHuman decision: REJECTED")
    return False


# ========================================
# TOOL EXECUTION
# ========================================

def execute_action(operation):

    action = operation["action"]
    customer = operation["customer"]
    amount = operation["amount"]

    print("\n========================================")
    print("TOOL EXECUTION")
    print("========================================")

    if action == "refund":

        print(
            f"Executing refund of ₹{amount:,.2f}"
        )

        print(f"Customer: {customer}")

        return {
            "status": "SUCCESS",
            "action": action,
            "customer": customer,
            "amount": amount
        }

    if action == "view_customer":

        print(f"Retrieving customer: {customer}")

        return {
            "status": "SUCCESS",
            "action": action,
            "customer": customer
        }

    if action == "view_order":

        print(f"Retrieving order information for: {customer}")

        return {
            "status": "SUCCESS",
            "action": action,
            "customer": customer
        }

    # This should never be reached because
    # the guardrail checks allowed actions.
    return {
        "status": "BLOCKED",
        "reason": "Unexpected action reached execution layer"
    }


# ========================================
# AGENT WORKFLOW
# ========================================

def run_agent(user_request):

    print("\n========================================")
    print("LESSON 44 — GUARDRAILS")
    print("========================================")

    print("\n========================================")
    print("AGENT REQUEST")
    print("========================================")

    print(user_request)

    # ------------------------------------
    # Step 1 — LLM interprets request
    # ------------------------------------

    llm_response = interpret_request(user_request)

    # ------------------------------------
    # Step 2 — Parse and validate
    # ------------------------------------

    try:

        operation = parse_operation(llm_response)

    except ValueError as e:

        print("\n========================================")
        print("LLM OUTPUT VALIDATION FAILED")
        print("========================================")

        print(f"Reason: {e}")
        print("Action blocked because the LLM response was invalid.")

        return {
            "status": "BLOCKED",
            "reason": str(e)
        }

    # ------------------------------------
    # Step 3 — Guardrail policy check
    # ------------------------------------

    guardrail_result = check_guardrail(operation)

    print("\n========================================")
    print("GUARDRAIL DECISION")
    print("========================================")

    print(f"Decision : {guardrail_result['decision']}")
    print(f"Reason   : {guardrail_result['reason']}")

    # ------------------------------------
    # BLOCK
    # ------------------------------------

    if guardrail_result["decision"] == "BLOCK":

        print("\nAction BLOCKED by guardrail.")

        return {
            "status": "BLOCKED",
            "reason": guardrail_result["reason"]
        }

    # ------------------------------------
    # HITL
    # ------------------------------------

    if guardrail_result["decision"] == "HITL":

        approved = request_human_approval(operation)

        if not approved:

            print("\nAction stopped because human approval was denied.")

            return {
                "status": "REJECTED",
                "reason": "Human approval denied"
            }

    # ------------------------------------
    # ALLOW / APPROVED
    # ------------------------------------

    result = execute_action(operation)

    print("\nFinal result:")
    print(result)

    return result


# ========================================
# MAIN
# ========================================

if __name__ == "__main__":

    # ------------------------------------
    # Test 1 — Safe action
    # ------------------------------------

    run_agent(
        "Refund ₹5,000 to customer C1001"
    )

    print("\n\n")

    # ------------------------------------
    # Test 2 — High-value action
    # ------------------------------------

    run_agent(
        "Refund ₹25,000 to customer C1001"
    )

    print("\n\n")

    # ------------------------------------
    # Test 3 — Prohibited action
    # ------------------------------------

    run_agent(
        "Delete the entire customer database"
    )

    print("\n========================================")
    print("LESSON 44 COMPLETE")
    print("========================================")