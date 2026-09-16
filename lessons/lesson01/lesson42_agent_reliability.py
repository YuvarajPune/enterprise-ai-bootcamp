# ============================================================
# LESSON 42 — AGENT RELIABILITY
#
# Integrated Experiment
#
# Demonstrates:
#   1. Structured errors
#   2. Error classification
#   3. Retry policy
#   4. Exponential backoff
#   5. Retry limits
#   6. Fallback capability
#   7. Fatal errors
#   8. Reliability decision gateway
#
# Architecture:
#
# Agent
#   ↓
# Tool Request
#   ↓
# Reliability Gateway
#   ↓
# Execute
#   ↓
# Error?
#   ↓
# Classify
#   ↓
# ┌──────────────┬──────────────┬─────────────┐
# │   RETRY      │   FALLBACK   │    FATAL    │
# │              │              │             │
# │ same tool   │ alternate    │ stop        │
# │ with backoff│ capability   │             │
# └──────────────┴──────────────┴─────────────┘
#
# Key principle:
#
# Reliability is a policy decision, not simply try/except.
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

from enum import Enum
import time
import json


# ============================================================
# 2. ERROR TYPES
#
# Structured errors allow the application to decide what to do.
# ============================================================


class ErrorType(Enum):

    TIMEOUT = "TIMEOUT"

    RATE_LIMITED = "RATE_LIMITED"

    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"

    NOT_FOUND = "NOT_FOUND"

    UNAUTHORIZED = "UNAUTHORIZED"

    INVALID_INPUT = "INVALID_INPUT"

    UNKNOWN = "UNKNOWN"


# ============================================================
# 3. STRUCTURED TOOL ERROR
# ============================================================


class ToolError(Exception):

    def __init__(
        self,
        error_type,
        message
    ):

        self.error_type = error_type

        self.message = message

        super().__init__(
            message
        )


# ============================================================
# 4. RELIABILITY POLICY
#
# This is application-controlled policy.
# ============================================================


RETRYABLE_ERRORS = {

    ErrorType.TIMEOUT,

    ErrorType.RATE_LIMITED,

    ErrorType.SERVICE_UNAVAILABLE
}


NON_RETRYABLE_ERRORS = {

    ErrorType.NOT_FOUND,

    ErrorType.UNAUTHORIZED,

    ErrorType.INVALID_INPUT
}


MAX_RETRIES = 3


# ============================================================
# 5. SIMULATED PRIMARY SERVICE
#
# First two attempts fail with TIMEOUT.
# Third attempt succeeds.
# ============================================================


primary_attempts = 0


def primary_customer_service(
    customer_id
):

    global primary_attempts

    primary_attempts += 1

    print(
        f"\n[PRIMARY] Attempt "
        f"{primary_attempts}"
    )

    if primary_attempts <= 2:

        raise ToolError(
            ErrorType.TIMEOUT,
            "Customer service timed out"
        )

    return {
        "customer_id": customer_id,
        "name": "ABC Industries",
        "status": "Active",
        "source": "Primary API"
    }


# ============================================================
# 6. SIMULATED BACKUP SERVICE
# ============================================================


def backup_customer_service(
    customer_id
):

    print(
        "\n[BACKUP] Executing backup service"
    )

    return {
        "customer_id": customer_id,
        "name": "ABC Industries",
        "status": "Active",
        "source": "Backup API"
    }


# ============================================================
# 7. ERROR CLASSIFICATION
#
# Converts a structured error into a reliability decision.
# ============================================================


def classify_error(
    error
):

    if error.error_type in RETRYABLE_ERRORS:

        return "RETRY"

    if error.error_type in NON_RETRYABLE_ERRORS:

        return "STOP"

    return "FALLBACK"


# ============================================================
# 8. EXPONENTIAL BACKOFF
#
# Wait time:
#
# Retry 1 → 1 second
# Retry 2 → 2 seconds
# Retry 3 → 4 seconds
#
# For the demo we keep the actual wait short.
# ============================================================


def calculate_backoff(
    retry_number
):

    return min(
        2 ** (retry_number - 1),
        4
    )


# ============================================================
# 9. RELIABILITY GATEWAY
#
# This is the main reliability control layer.
# ============================================================


def execute_with_reliability(
    primary_tool,
    fallback_tool,
    customer_id
):

    print(
        "\n" + "=" * 70
    )

    print(
        "RELIABILITY GATEWAY"
    )

    print(
        "=" * 70
    )

    retry_count = 0

    # --------------------------------------------------------
    # Initial execution
    # --------------------------------------------------------

    while True:

        try:

            result = primary_tool(
                customer_id
            )

            print(
                "\nPRIMARY EXECUTION: SUCCESS"
            )

            return {
                "success": True,
                "result": result,
                "strategy": "PRIMARY"
            }

        except ToolError as error:

            print(
                "\nERROR:"
            )

            print(
                json.dumps(
                    {
                        "type": error.error_type.value,
                        "message": error.message
                    },
                    indent=2
                )
            )

            decision = classify_error(
                error
            )

            print(
                f"RELIABILITY DECISION: "
                f"{decision}"
            )

            # =================================================
            # RETRY
            # =================================================

            if decision == "RETRY":

                if retry_count >= MAX_RETRIES:

                    print(
                        "\nMAX RETRIES REACHED"
                    )

                    print(
                        "Switching to fallback."
                    )

                    break

                retry_count += 1

                backoff = calculate_backoff(
                    retry_count
                )

                print(
                    f"Retry #{retry_count}"
                )

                print(
                    f"Backoff: {backoff} seconds"
                )

                # ------------------------------------------------
                # Keep demo fast.
                # We simulate backoff rather than actually waiting
                # the full duration.
                # ------------------------------------------------

                time.sleep(0.5)

                continue

            # =================================================
            # STOP
            # =================================================

            elif decision == "STOP":

                print(
                    "\nNON-RETRYABLE ERROR"
                )

                print(
                    "Execution stopped."
                )

                return {
                    "success": False,
                    "result": None,
                    "strategy": "STOP",
                    "error": error.error_type.value
                }

            # =================================================
            # FALLBACK
            # =================================================

            elif decision == "FALLBACK":

                print(
                    "\nSwitching to fallback."
                )

                break

    # ========================================================
    # FALLBACK EXECUTION
    # ========================================================

    try:

        result = fallback_tool(
            customer_id
        )

        print(
            "\nFALLBACK EXECUTION: SUCCESS"
        )

        return {
            "success": True,
            "result": result,
            "strategy": "FALLBACK"
        }

    except ToolError as error:

        print(
            "\nFALLBACK FAILED:"
        )

        print(
            error.message
        )

        return {
            "success": False,
            "result": None,
            "strategy": "FALLBACK_FAILED",
            "error": error.error_type.value
        }


# ============================================================
# 10. MAIN
# ============================================================


def main():

    customer_id = "C123"

    print(
        "\n" + "=" * 70
    )

    print(
        "LESSON 42 — AGENT RELIABILITY"
    )

    print(
        "=" * 70
    )

    print(
        f"\nCustomer: {customer_id}"
    )

    # --------------------------------------------------------
    # Execute through reliability gateway.
    # --------------------------------------------------------

    result = execute_with_reliability(
        primary_customer_service,
        backup_customer_service,
        customer_id
    )

    # --------------------------------------------------------
    # Final result.
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL RESULT"
    )

    print(
        "=" * 70
    )

    print(
        json.dumps(
            result,
            indent=2
        )
    )

    if result["success"]:

        print(
            "\nWORKFLOW: COMPLETED"
        )

    else:

        print(
            "\nWORKFLOW: FAILED"
        )


# ============================================================
# 11. APPLICATION ENTRY POINT
# ============================================================


if __name__ == "__main__":

    main()