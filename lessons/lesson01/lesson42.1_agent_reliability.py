"""
Lesson 42.1 — Agent Reliability V2

Goal:
    Extend Lesson 42 reliability with:
    1. LLM-generated structured operation request
    2. Transient failure -> retry -> success
    3. Permanent failure -> stop immediately
    4. Retry exhaustion -> fallback
    5. Fallback failure -> controlled failure

Architecture:

    LLM
      |
      | Structured operation request
      v
    Reliability Gateway
      |
      +--> Error Classification
      +--> Retry Policy
      +--> Backoff
      +--> Retry Limit
      +--> Fallback
      |
      v
    Enterprise Tool

Key principle:
    LLM decides WHAT operation is needed.
    Application decides HOW failures are handled.

Setup:
    pip install google-genai python-dotenv

Environment:
    GEMINI_API_KEY=your_key_here

Run:
    python lesson42.1_agent_reliability.py
"""

import os
import time
from enum import Enum

from dotenv import load_dotenv
from google import genai


# ============================================================
# 1. LLM SETUP
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set.")

client = genai.Client(api_key=api_key)


# ============================================================
# 2. STRUCTURED ERROR TYPES
# ============================================================

class ErrorType(Enum):
    TIMEOUT = "TIMEOUT"
    RATE_LIMITED = "RATE_LIMITED"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    INVALID_INPUT = "INVALID_INPUT"
    UNKNOWN = "UNKNOWN"


class ToolError(Exception):

    def __init__(self, error_type, message):
        self.error_type = error_type
        self.message = message
        super().__init__(message)


# ============================================================
# 3. RELIABILITY POLICY
# ============================================================

RETRYABLE_ERRORS = {
    ErrorType.TIMEOUT,
    ErrorType.RATE_LIMITED,
    ErrorType.SERVICE_UNAVAILABLE,
}

NON_RETRYABLE_ERRORS = {
    ErrorType.NOT_FOUND,
    ErrorType.UNAUTHORIZED,
    ErrorType.INVALID_INPUT,
}


MAX_RETRIES = 2


def classify_error(error):
    """
    Convert a tool failure into a structured error type.
    """

    if isinstance(error, ToolError):
        return error.error_type

    return ErrorType.UNKNOWN


def calculate_backoff(retry_number):
    """
    Exponential backoff.

    Retry 1 -> 1 second
    Retry 2 -> 2 seconds
    Retry 3 -> 4 seconds
    """

    return min(2 ** (retry_number - 1), 4)


# ============================================================
# 4. LLM REQUEST GENERATION
# ============================================================

def generate_agent_request():

    prompt = """
You are an enterprise AI agent.

The user wants to retrieve customer information.

Return ONLY valid JSON in this exact structure:

{
    "operation": "get_customer",
    "customer_id": "C1001"
}
"""

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt
    )

    print("\nLLM generated request:")
    print(interaction.output_text)

    return interaction.output_text


# ============================================================
# 5. SIMULATED ENTERPRISE SERVICES
# ============================================================

class PrimaryCustomerService:

    def __init__(self):
        self.call_count = 0

    def get_customer(self, customer_id):

        self.call_count += 1

        print(
            f"\nPrimary service call #{self.call_count}"
        )

        # First two calls fail with transient error.
        if self.call_count <= 2:
            raise ToolError(
                ErrorType.TIMEOUT,
                "Primary customer service timed out."
            )

        return {
            "source": "PRIMARY_API",
            "customer_id": customer_id,
            "name": "ABC Industries",
            "status": "Active"
        }


class PermanentFailureService:

    def get_customer(self, customer_id):

        print("\nPermanent-failure service called.")

        raise ToolError(
            ErrorType.NOT_FOUND,
            f"Customer {customer_id} was not found."
        )


class ExhaustedPrimaryService:

    def __init__(self):
        self.call_count = 0

    def get_customer(self, customer_id):

        self.call_count += 1

        print(
            f"\nExhausted primary service call #{self.call_count}"
        )

        raise ToolError(
            ErrorType.SERVICE_UNAVAILABLE,
            "Primary service remains unavailable."
        )


class BackupCustomerService:

    def get_customer(self, customer_id):

        print("\nBackup customer service called.")

        return {
            "source": "BACKUP_API",
            "customer_id": customer_id,
            "name": "ABC Industries",
            "status": "Active"
        }


class FailedBackupCustomerService:

    def get_customer(self, customer_id):

        print("\nBackup customer service called.")

        raise ToolError(
            ErrorType.SERVICE_UNAVAILABLE,
            "Backup service is also unavailable."
        )


# ============================================================
# 6. RELIABILITY GATEWAY
# ============================================================

def execute_with_reliability(
    operation,
    customer_id,
    primary_service,
    fallback_service=None
):
    """
    Reliability Gateway.

    Responsibilities:
        - Execute tool
        - Classify errors
        - Retry transient errors
        - Stop on permanent errors
        - Use fallback after retry exhaustion
        - Return controlled result
    """

    print("\n========================================")
    print("RELIABILITY GATEWAY")
    print("========================================")

    retry_count = 0

    while True:

        try:

            print(
                f"\nExecuting operation: {operation}"
            )

            result = primary_service.get_customer(
                customer_id
            )

            print("\nSUCCESS")
            print("Primary execution succeeded.")

            return result

        except Exception as error:

            error_type = classify_error(error)

            print(
                f"\nERROR: {error_type.value}"
            )

            print(
                f"Message: {error}"
            )

            # ------------------------------------------------
            # Permanent error
            # ------------------------------------------------

            if error_type in NON_RETRYABLE_ERRORS:

                print(
                    "\nPOLICY: NON-RETRYABLE ERROR"
                )

                print(
                    "Action: STOP"
                )

                return {
                    "status": "FAILED",
                    "reason": error_type.value,
                    "message": str(error)
                }

            # ------------------------------------------------
            # Retryable error
            # ------------------------------------------------

            if error_type in RETRYABLE_ERRORS:

                if retry_count < MAX_RETRIES:

                    retry_count += 1

                    backoff = calculate_backoff(
                        retry_count
                    )

                    print(
                        "\nPOLICY: RETRYABLE ERROR"
                    )

                    print(
                        f"Action: RETRY #{retry_count}"
                    )

                    print(
                        f"Conceptual backoff: "
                        f"{backoff} second(s)"
                    )

                    # Keep the demonstration fast.
                    time.sleep(0.5)

                    continue

                # --------------------------------------------
                # Retry exhausted
                # --------------------------------------------

                print(
                    "\nRETRY LIMIT EXCEEDED"
                )

                if fallback_service:

                    print(
                        "Action: FALLBACK"
                    )

                    try:

                        result = fallback_service.get_customer(
                            customer_id
                        )

                        print(
                            "\nFALLBACK SUCCESS"
                        )

                        return result

                    except Exception as fallback_error:

                        fallback_type = classify_error(
                            fallback_error
                        )

                        print(
                            "\nFALLBACK FAILED"
                        )

                        print(
                            f"Fallback error: "
                            f"{fallback_type.value}"
                        )

                        return {
                            "status": "FAILED",
                            "reason": "FALLBACK_FAILED",
                            "primary_error": error_type.value,
                            "fallback_error": fallback_type.value
                        }

                print(
                    "No fallback available."
                )

                return {
                    "status": "FAILED",
                    "reason": "RETRY_EXHAUSTED"
                }

            # ------------------------------------------------
            # Unknown error
            # ------------------------------------------------

            print(
                "\nPOLICY: UNKNOWN ERROR"
            )

            print(
                "Action: STOP"
            )

            return {
                "status": "FAILED",
                "reason": "UNKNOWN_ERROR",
                "message": str(error)
            }


# ============================================================
# 7. SCENARIO 1
# ============================================================

def scenario_transient_success(customer_id):

    print("\n\n")
    print("########################################")
    print("SCENARIO 1")
    print("Transient failure -> retry -> success")
    print("########################################")

    primary = PrimaryCustomerService()

    result = execute_with_reliability(
        operation="get_customer",
        customer_id=customer_id,
        primary_service=primary
    )

    print("\nFinal result:")
    print(result)


# ============================================================
# 8. SCENARIO 2
# ============================================================

def scenario_permanent_failure(customer_id):

    print("\n\n")
    print("########################################")
    print("SCENARIO 2")
    print("Permanent failure -> no retry -> stop")
    print("########################################")

    primary = PermanentFailureService()

    result = execute_with_reliability(
        operation="get_customer",
        customer_id=customer_id,
        primary_service=primary
    )

    print("\nFinal result:")
    print(result)


# ============================================================
# 9. SCENARIO 3
# ============================================================

def scenario_retry_exhaustion(customer_id):

    print("\n\n")
    print("########################################")
    print("SCENARIO 3")
    print("Retry exhaustion -> fallback")
    print("########################################")

    primary = ExhaustedPrimaryService()
    backup = BackupCustomerService()

    result = execute_with_reliability(
        operation="get_customer",
        customer_id=customer_id,
        primary_service=primary,
        fallback_service=backup
    )

    print("\nFinal result:")
    print(result)


# ============================================================
# 10. SCENARIO 4
# ============================================================

def scenario_fallback_failure(customer_id):

    print("\n\n")
    print("########################################")
    print("SCENARIO 4")
    print("Primary failure -> retries -> fallback failure")
    print("########################################")

    primary = ExhaustedPrimaryService()
    backup = FailedBackupCustomerService()

    result = execute_with_reliability(
        operation="get_customer",
        customer_id=customer_id,
        primary_service=primary,
        fallback_service=backup
    )

    print("\nFinal result:")
    print(result)


# ============================================================
# 11. MAIN AGENT FLOW
# ============================================================

def main():

    print("========================================")
    print("LESSON 42.1 — AGENT RELIABILITY V2")
    print("========================================")

    # --------------------------------------------------------
    # LLM determines WHAT the agent wants to do.
    # --------------------------------------------------------

    request_json = generate_agent_request()

    # For this controlled learning experiment,
    # we know the LLM is expected to return C1001.
    # In production, validate and parse the JSON
    # before execution.

    customer_id = "C1001"

    print(
        f"\nAgent operation: get_customer"
    )

    print(
        f"Customer ID: {customer_id}"
    )

    # --------------------------------------------------------
    # Reliability experiments
    # --------------------------------------------------------

    scenario_transient_success(customer_id)

    scenario_permanent_failure(customer_id)

    scenario_retry_exhaustion(customer_id)

    scenario_fallback_failure(customer_id)

    print("\n\n")
    print("========================================")
    print("LESSON 42.1 COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()