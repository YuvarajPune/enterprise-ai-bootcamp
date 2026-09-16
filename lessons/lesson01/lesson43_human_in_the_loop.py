"""
Lesson 43 — Human-in-the-Loop & Approvals

Goal:
    Demonstrate how an enterprise agent can pause for human
    approval before executing a high-risk action.

Architecture:

    User Request
         |
         v
        LLM
         |
         | decides WHAT operation is needed
         v
    Policy Engine
         |
         +---- Low Risk ------> Execute
         |
         +---- High Risk -----> Human Approval
                                      |
                               +------+------+
                               |             |
                           APPROVE        REJECT
                               |             |
                            Execute          Stop

Important principle:
    LLM decides WHAT.
    Application policy decides WHETHER approval is required.
    Human decides whether to authorize the high-risk action.

Setup:
    pip install google-genai python-dotenv

Environment:
    GEMINI_API_KEY=your_key_here

Run:
    python lessons/lesson01/lesson43_human_in_the_loop.py
"""

import os
import re

from dotenv import load_dotenv
from google import genai


# ============================================================
# 1. GEMINI SETUP
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set.")

client = genai.Client(api_key=api_key)


# ============================================================
# 2. LLM — DETERMINE THE BUSINESS OPERATION
# ============================================================

def understand_request(user_request):
    """
    Ask the LLM to identify the business operation.

    The LLM determines WHAT the user wants.

    It does NOT decide whether approval is required.
    """

    prompt = f"""
You are an enterprise AI agent.

Analyze the following user request.

Return ONLY one line in this exact format:

ACTION=<action>;CUSTOMER=<customer_id>;AMOUNT=<amount>

Allowed actions:
- refund
- view_order

If amount is not applicable, use 0.

User request:
{user_request}
"""

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt
    )

    result = interaction.output_text.strip()

    print("\nLLM interpretation:")
    print(result)

    return result


# ============================================================
# 3. PARSE AND VALIDATE LLM OUTPUT
# ============================================================

def parse_request(llm_output):

    pattern = (
        r"ACTION=(.*?);"
        r"CUSTOMER=(.*?);"
        r"AMOUNT=(.*)"
    )

    match = re.search(pattern, llm_output)

    if not match:
        raise ValueError(
            "Invalid LLM output format."
        )

    action = match.group(1).strip()
    customer = match.group(2).strip()
    amount = float(match.group(3).strip())

    allowed_actions = {
        "refund",
        "view_order"
    }

    if action not in allowed_actions:
        raise ValueError(
            f"Unauthorized action: {action}"
        )

    return {
        "action": action,
        "customer": customer,
        "amount": amount
    }


# ============================================================
# 4. ENTERPRISE APPROVAL POLICY
# ============================================================

REFUND_APPROVAL_LIMIT = 10000


def requires_approval(request):
    """
    Deterministic enterprise policy.

    The LLM does NOT control this decision.
    """

    action = request["action"]
    amount = request["amount"]

    if action == "refund" and amount > REFUND_APPROVAL_LIMIT:
        return True

    return False


# ============================================================
# 5. HUMAN APPROVAL GATE
# ============================================================

def request_human_approval(request):

    print("\n========================================")
    print("HUMAN APPROVAL REQUIRED")
    print("========================================")

    print(
        f"Action   : {request['action']}"
    )

    print(
        f"Customer : {request['customer']}"
    )

    print(
        f"Amount   : ₹{request['amount']:,.2f}"
    )

    print(
        "\nThis action requires human authorization."
    )

    while True:

        decision = input(
            "\nApprove this action? (yes/no): "
        ).strip().lower()

        if decision in {"yes", "y"}:
            print("\nHuman decision: APPROVED")
            return True

        if decision in {"no", "n"}:
            print("\nHuman decision: REJECTED")
            return False

        print(
            "Please enter yes or no."
        )


# ============================================================
# 6. ENTERPRISE TOOL EXECUTION
# ============================================================

def execute_action(request):

    print("\n========================================")
    print("TOOL EXECUTION")
    print("========================================")

    action = request["action"]

    if action == "refund":

        print(
            f"Executing refund of "
            f"₹{request['amount']:,.2f}"
        )

        print(
            f"Customer: {request['customer']}"
        )

        return {
            "status": "SUCCESS",
            "action": "refund",
            "customer": request["customer"],
            "amount": request["amount"]
        }

    if action == "view_order":

        print(
            f"Retrieving orders for "
            f"{request['customer']}"
        )

        return {
            "status": "SUCCESS",
            "action": "view_order",
            "customer": request["customer"]
        }

    raise ValueError(
        f"Unsupported action: {action}"
    )


# ============================================================
# 7. AGENT WORKFLOW
# ============================================================

def run_agent(user_request):

    print("\n========================================")
    print("AGENT REQUEST")
    print("========================================")

    print(user_request)

    # --------------------------------------------------------
    # Step 1 — LLM determines WHAT is required
    # --------------------------------------------------------

    llm_output = understand_request(
        user_request
    )

    # --------------------------------------------------------
    # Step 2 — Validate LLM output
    # --------------------------------------------------------

    request = parse_request(
        llm_output
    )

    print("\nValidated operation:")
    print(request)

    # --------------------------------------------------------
    # Step 3 — Enterprise policy check
    # --------------------------------------------------------

    approval_required = requires_approval(
        request
    )

    print(
        f"\nApproval required: "
        f"{approval_required}"
    )

    # --------------------------------------------------------
    # Step 4 — Human approval if required
    # --------------------------------------------------------

    if approval_required:

        approved = request_human_approval(
            request
        )

        if not approved:

            print("\nACTION BLOCKED")

            return {
                "status": "REJECTED",
                "reason": "Human approval denied"
            }

    else:

        print(
            "\nNo human approval required."
        )

    # --------------------------------------------------------
    # Step 5 — Execute
    # --------------------------------------------------------

    result = execute_action(
        request
    )

    print("\nFinal result:")
    print(result)

    return result


# ============================================================
# 8. MAIN
# ============================================================

def main():

    print("========================================")
    print("LESSON 43 — HUMAN-IN-THE-LOOP")
    print("========================================")

    user_request = (
        "Refund ₹25,000 to customer C1001"
    )

    run_agent(user_request)

    print("\n========================================")
    print("LESSON 43 COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()