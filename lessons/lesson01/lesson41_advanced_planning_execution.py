# ============================================================
# LESSON 41 — ADVANCED PLANNING & EXECUTION
#
# Integrated Production-Style Experiment
#
# Demonstrates:
#   1. LLM-based planning
#   2. Application plan validation
#   3. Dependency analysis
#   4. State-aware execution
#   5. Parallel execution of independent steps
#   6. Dynamic re-planning after failure
#   7. Preservation of successful execution state
#   8. Recovery-plan validation
#   9. Final state validation
#
# Architecture:
#
# User Goal
#     ↓
# LLM Planner
#     ↓
# Structured Plan
#     ↓
# Plan Validation
#     ↓
# Dependency + State Validation
#     ↓
# Plan Executor
#     ↓
# Enterprise Tools
#     ↓
# State
#     ↓
# Failure?
#     ↓
# LLM Re-planner
#     ↓
# Revised Plan
#     ↓
# Validate Again
#     ↓
# Execute Again
#     ↓
# Final State Validation
#     ↓
# Final Result
#
# Key principle:
#
# LLM = plans / reasons
# Application = controls / validates
# Executor = performs
# Tools = enterprise capabilities
# State = carries execution results
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

from dotenv import load_dotenv
from google import genai
from google.genai import types

import os
import json
from concurrent.futures import ThreadPoolExecutor


# ============================================================
# 2. SETUP
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY not found in environment."
    )

client = genai.Client(
    api_key=api_key
)


# ============================================================
# 3. AVAILABLE ACTIONS
#
# These are the capabilities that the application exposes.
#
# The LLM is allowed to choose only from these actions.
# ============================================================

ALLOWED_ACTIONS = {
    "get_customer",
    "get_orders",
    "get_support_tickets",
    "calculate_total",
    "summarize",
    "backup_get_orders"
}


# Maximum number of steps allowed in one plan.

MAX_STEPS = 8


# ============================================================
# 4. REQUIRED STATE
#
# Defines what information must already exist before an action
# can execute.
#
# This is APPLICATION CONTROL LOGIC.
#
# The LLM does not define these requirements.
# ============================================================

REQUIRED_STATE = {

    "get_customer": [],

    "get_orders": [],

    "backup_get_orders": [],

    "get_support_tickets": [],

    "calculate_total": [
        "orders"
    ],

    "summarize": [
        "customer",
        "total",
        "tickets"
    ]
}


# ============================================================
# 5. MOCK ENTERPRISE TOOLS
#
# These simulate real enterprise systems.
# ============================================================


def get_customer(customer_id):

    print(
        f"\n[TOOL] get_customer({customer_id})"
    )

    return {
        "customer_id": customer_id,
        "name": "ABC Industries",
        "status": "Active"
    }


def get_orders(customer_id):

    print(
        f"\n[TOOL] get_orders({customer_id})"
    )

    # --------------------------------------------------------
    # Deliberately fail.
    #
    # This allows us to demonstrate dynamic re-planning.
    # --------------------------------------------------------

    raise ConnectionError(
        "Primary order API is unavailable"
    )


def backup_get_orders(customer_id):

    print(
        f"\n[TOOL] backup_get_orders({customer_id})"
    )

    return [
        {
            "order_id": "O101",
            "amount": 12000
        },
        {
            "order_id": "O102",
            "amount": 18000
        },
        {
            "order_id": "O103",
            "amount": 10000
        }
    ]


def get_support_tickets(customer_id):

    print(
        f"\n[TOOL] get_support_tickets({customer_id})"
    )

    return [
        {
            "ticket_id": "T101",
            "status": "Open",
            "priority": "High"
        },
        {
            "ticket_id": "T102",
            "status": "Closed",
            "priority": "Low"
        }
    ]


def calculate_total(orders):

    print(
        "\n[TOOL] calculate_total()"
    )

    return sum(
        order["amount"]
        for order in orders
    )


def summarize(
    customer,
    total,
    tickets
):

    print(
        "\n[TOOL] summarize()"
    )

    open_tickets = [
        ticket
        for ticket in tickets
        if ticket["status"] == "Open"
    ]

    return {
        "customer": customer["name"],
        "status": customer["status"],
        "total_order_value": total,
        "open_support_tickets": len(
            open_tickets
        )
    }


# ============================================================
# 6. LLM PLANNER
#
# The LLM creates a structured plan.
#
# It proposes what should happen.
# It does not execute anything.
# ============================================================


def create_plan(user_goal):

    print(
        "\n" + "=" * 70
    )

    print(
        "LLM PLANNER"
    )

    print(
        "=" * 70
    )

    planning_prompt = f"""
You are an AI planning agent.

Convert the user's goal into a structured execution plan.

Available actions:

{sorted(ALLOWED_ACTIONS)}

Important rules:

1. Only use actions from the available actions list.
2. Each step must have a unique integer ID.
3. Use depends_on to represent dependencies.
4. A step should depend on another step when it requires
   information produced by that step.
5. Do not execute any action.
6. Return ONLY valid JSON.

Use exactly this structure:

{{
    "goal": "short description",
    "steps": [
        {{
            "id": 1,
            "action": "action name",
            "description": "what this step does",
            "depends_on": []
        }}
    ]
}}

User goal:

{user_goal}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=planning_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    # --------------------------------------------------------
    # Never blindly trust LLM output.
    # --------------------------------------------------------

    try:

        plan = json.loads(
            response.text
        )

    except json.JSONDecodeError:

        print(
            "\nPLANNING FAILED"
        )

        print(
            "LLM did not return valid JSON."
        )

        print(
            "\nRaw response:"
        )

        print(
            response.text
        )

        raise

    print(
        "\nGenerated Plan:"
    )

    print(
        json.dumps(
            plan,
            indent=2
        )
    )

    return plan


# ============================================================
# 7. PLAN VALIDATION
#
# Application validates the LLM-generated plan.
# ============================================================


def validate_plan(plan):

    print(
        "\n" + "=" * 70
    )

    print(
        "PLAN VALIDATION"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Basic structure
    # --------------------------------------------------------

    if "goal" not in plan:

        return False, "Missing goal"

    if "steps" not in plan:

        return False, "Missing steps"

    steps = plan["steps"]

    if not isinstance(
        steps,
        list
    ):

        return False, (
            "Steps must be a list"
        )

    if not steps:

        return False, (
            "Plan contains no steps"
        )

    # --------------------------------------------------------
    # Plan size
    # --------------------------------------------------------

    if len(steps) > MAX_STEPS:

        return False, (
            "Plan exceeds maximum "
            "step limit"
        )

    # --------------------------------------------------------
    # Validate IDs and actions
    # --------------------------------------------------------

    step_ids = set()

    for step in steps:

        if "id" not in step:

            return False, (
                "Step is missing ID"
            )

        if "action" not in step:

            return False, (
                f"Step {step['id']} "
                f"is missing action"
            )

        step_id = step["id"]

        action = step["action"]

        # Unique step IDs

        if step_id in step_ids:

            return False, (
                f"Duplicate step ID: "
                f"{step_id}"
            )

        step_ids.add(
            step_id
        )

        # Allowed action

        if action not in ALLOWED_ACTIONS:

            return False, (
                f"Action not allowed: "
                f"{action}"
            )

    # --------------------------------------------------------
    # Validate dependencies
    # --------------------------------------------------------

    for step in steps:

        dependencies = step.get(
            "depends_on",
            []
        )

        if not isinstance(
            dependencies,
            list
        ):

            return False, (
                f"Invalid dependencies "
                f"for step {step['id']}"
            )

        for dependency in dependencies:

            if dependency not in step_ids:

                return False, (
                    f"Invalid dependency "
                    f"{dependency}"
                )

            if dependency == step["id"]:

                return False, (
                    f"Step {step['id']} "
                    f"cannot depend on itself"
                )

    print(
        "VALIDATION: PASSED"
    )

    return True, (
        "Plan validation passed"
    )


# ============================================================
# 8. STATE VALIDATION
#
# This is the additional protection we discovered during the
# first execution.
#
# A plan can be valid but the CURRENT STATE may still be
# insufficient to execute a step.
# ============================================================


def validate_step_state(
    step,
    state
):

    action = step["action"]

    required_state = REQUIRED_STATE.get(
        action,
        []
    )

    for required_item in required_state:

        if required_item not in state:

            return False, (
                f"Missing required state: "
                f"{required_item}"
            )

    return True, (
        "State validation passed"
    )


# ============================================================
# 9. DEPENDENCY ANALYSIS
#
# Find steps whose dependencies have completed.
# ============================================================


def get_ready_steps(
    plan,
    completed_steps
):

    ready = []

    for step in plan["steps"]:

        step_id = step["id"]

        if step_id in completed_steps:

            continue

        dependencies = step.get(
            "depends_on",
            []
        )

        if all(
            dependency in completed_steps
            for dependency in dependencies
        ):

            ready.append(
                step
            )

    return ready


# ============================================================
# 10. TOOL EXECUTION DISPATCHER
#
# Maps a planned action to an actual application function.
# ============================================================


def execute_action(
    action,
    customer_id,
    state
):

    if action == "get_customer":

        return get_customer(
            customer_id
        )

    elif action == "get_orders":

        return get_orders(
            customer_id
        )

    elif action == "backup_get_orders":

        return backup_get_orders(
            customer_id
        )

    elif action == "get_support_tickets":

        return get_support_tickets(
            customer_id
        )

    elif action == "calculate_total":

        return calculate_total(
            state["orders"]
        )

    elif action == "summarize":

        return summarize(
            state["customer"],
            state["total"],
            state["tickets"]
        )

    else:

        raise ValueError(
            f"Unknown action: {action}"
        )


# ============================================================
# 11. APPLY RESULT TO STATE
#
# Tool results become state for subsequent steps.
# ============================================================


def apply_result(
    step,
    result,
    state
):

    action = step["action"]

    if action == "get_customer":

        state["customer"] = result

    elif action in {
        "get_orders",
        "backup_get_orders"
    }:

        state["orders"] = result

    elif action == "get_support_tickets":

        state["tickets"] = result

    elif action == "calculate_total":

        state["total"] = result

    elif action == "summarize":

        state["summary"] = result

    print(
        f"Step {step['id']} completed."
    )


# ============================================================
# 12. PLAN EXECUTOR
#
# Important:
#
# The executor:
#   - checks dependencies
#   - checks required state
#   - executes independent steps in parallel
#   - preserves successful state
#   - stops safely on failure
# ============================================================


def execute_plan(
    plan,
    customer_id,
    existing_state=None
):

    print(
        "\n" + "=" * 70
    )

    print(
        "PLAN EXECUTION"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Preserve state from previous execution.
    #
    # This is important during recovery.
    # --------------------------------------------------------

    if existing_state is not None:

        state = dict(
            existing_state
        )

    else:

        state = {}

    completed_steps = set()

    # --------------------------------------------------------
    # Mark steps as already satisfied if the required state
    # already exists.
    #
    # This allows a recovery plan to continue from the current
    # state instead of starting everything from zero.
    # --------------------------------------------------------

    for step in plan["steps"]:

        action = step["action"]

        required = REQUIRED_STATE.get(
            action,
            []
        )

        # We don't automatically mark the step complete merely
        # because its output exists. Recovery plans are allowed
        # to contain explicit actions that need to run.
        #
        # Therefore this section only preserves the state,
        # while completed_steps starts empty.

        _ = required

    # --------------------------------------------------------
    # Main execution loop
    # --------------------------------------------------------

    while len(
        completed_steps
    ) < len(plan["steps"]):

        ready_steps = get_ready_steps(
            plan,
            completed_steps
        )

        if not ready_steps:

            print(
                "\nEXECUTION STOPPED:"
            )

            print(
                "No executable steps remain."
            )

            return {
                "success": False,
                "state": state,
                "failed_step": None,
                "error": (
                    "Dependency deadlock"
                )
            }

        print(
            f"\nReady steps: "
            f"{[step['id'] for step in ready_steps]}"
        )

        # ====================================================
        # Validate state BEFORE execution
        # ====================================================

        executable_steps = []

        for step in ready_steps:

            valid, message = (
                validate_step_state(
                    step,
                    state
                )
            )

            if not valid:

                print(
                    f"\nSTEP {step['id']} "
                    f"BLOCKED:"
                )

                print(
                    message
                )

                return {
                    "success": False,
                    "state": state,
                    "failed_step": step,
                    "error": message
                }

            executable_steps.append(
                step
            )

        # ====================================================
        # PARALLEL EXECUTION
        # ====================================================

        if len(executable_steps) > 1:

            print(
                "Multiple independent "
                "steps detected."
            )

            print(
                "Executing ready steps "
                "in parallel."
            )

            with ThreadPoolExecutor(
                max_workers=len(
                    executable_steps
                )
            ) as executor:

                futures = {}

                for step in executable_steps:

                    future = executor.submit(
                        execute_action,
                        step["action"],
                        customer_id,
                        state
                    )

                    futures[
                        step["id"]
                    ] = (
                        step,
                        future
                    )

                # ------------------------------------------------
                # Collect results.
                # ------------------------------------------------

                for step_id, (
                    step,
                    future
                ) in futures.items():

                    try:

                        result = future.result()

                        apply_result(
                            step,
                            result,
                            state
                        )

                        completed_steps.add(
                            step_id
                        )

                    except Exception as error:

                        print(
                            f"\nSTEP {step_id} "
                            f"FAILED:"
                        )

                        print(
                            str(error)
                        )

                        return {
                            "success": False,
                            "state": state,
                            "failed_step": step,
                            "error": str(error)
                        }

        # ====================================================
        # SEQUENTIAL EXECUTION
        # ====================================================

        else:

            step = executable_steps[0]

            print(
                f"\nExecuting Step "
                f"{step['id']}: "
                f"{step['action']}"
            )

            try:

                result = execute_action(
                    step["action"],
                    customer_id,
                    state
                )

                apply_result(
                    step,
                    result,
                    state
                )

                completed_steps.add(
                    step["id"]
                )

            except Exception as error:

                print(
                    f"\nSTEP {step['id']} "
                    f"FAILED:"
                )

                print(
                    str(error)
                )

                return {
                    "success": False,
                    "state": state,
                    "failed_step": step,
                    "error": str(error)
                }

    return {
        "success": True,
        "state": state,
        "failed_step": None,
        "error": None
    }


# ============================================================
# 13. DYNAMIC RE-PLANNER
#
# The LLM receives:
#
#   - original plan
#   - failed step
#   - error
#   - current state
#
# It then proposes a revised plan.
# ============================================================


def create_recovery_plan(
    original_plan,
    failed_step,
    error,
    state
):

    print(
        "\n" + "=" * 70
    )

    print(
        "DYNAMIC RE-PLANNING"
    )

    print(
        "=" * 70
    )

    recovery_prompt = f"""
You are an AI recovery planner.

The original plan was:

{json.dumps(original_plan, indent=2)}

The following step failed:

{json.dumps(failed_step, indent=2)}

Error:

{error}

Current execution state:

{json.dumps(state, indent=2)}

Available actions:

{sorted(ALLOWED_ACTIONS)}

Important rules:

1. Continue toward the original goal.
2. Consider the current execution state.
3. Do not assume missing information exists.
4. If information required by a future step is missing,
   include the appropriate action to obtain it.
5. Use backup_get_orders instead of get_orders because
   the primary order API has failed.
6. Only use actions from the available actions list.
7. Use dependencies correctly.
8. Return ONLY valid JSON.

Use this structure:

{{
    "goal": "short description",
    "steps": [
        {{
            "id": 1,
            "action": "action name",
            "description": "what this step does",
            "depends_on": []
        }}
    ]
}}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=recovery_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    try:

        recovery_plan = json.loads(
            response.text
        )

    except json.JSONDecodeError:

        print(
            "\nRECOVERY PLANNING FAILED"
        )

        print(
            "LLM did not return valid JSON."
        )

        print(
            "\nRaw response:"
        )

        print(
            response.text
        )

        raise

    print(
        "\nRevised Plan:"
    )

    print(
        json.dumps(
            recovery_plan,
            indent=2
        )
    )

    return recovery_plan


# ============================================================
# 14. FINAL STATE VALIDATION
#
# Before declaring success, verify that the expected business
# outputs actually exist.
# ============================================================


def validate_final_state(
    state
):

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL STATE VALIDATION"
    )

    print(
        "=" * 70
    )

    required_final_state = [
        "customer",
        "orders",
        "tickets",
        "total",
        "summary"
    ]

    for item in required_final_state:

        if item not in state:

            print(
                f"FINAL VALIDATION FAILED: "
                f"Missing {item}"
            )

            return False

    print(
        "FINAL STATE VALIDATION: PASSED"
    )

    return True


# ============================================================
# 15. MAIN WORKFLOW
# ============================================================


def main():

    user_goal = """
    Prepare a customer report for customer C123.

    The report should:
    1. Get customer details.
    2. Get recent orders.
    3. Check support tickets.
    4. Calculate total order value.
    5. Summarize the findings.
    """

    print(
        "\n" + "=" * 70
    )

    print(
        "LESSON 41 — "
        "ADVANCED PLANNING & EXECUTION"
    )

    print(
        "=" * 70
    )

    print(
        "\nUSER GOAL:"
    )

    print(
        user_goal
    )

    # ========================================================
    # PHASE 1 — CREATE PLAN
    # ========================================================

    plan = create_plan(
        user_goal
    )

    # ========================================================
    # PHASE 2 — VALIDATE PLAN
    # ========================================================

    valid, message = validate_plan(
        plan
    )

    if not valid:

        print(
            f"\nVALIDATION FAILED: "
            f"{message}"
        )

        return

    # ========================================================
    # PHASE 3 — EXECUTE PLAN
    # ========================================================

    execution = execute_plan(
        plan,
        "C123"
    )

    # ========================================================
    # PHASE 4 — DYNAMIC RE-PLANNING
    # ========================================================

    if not execution["success"]:

        print(
            "\nINITIAL PLAN FAILED."
        )

        print(
            "\nPreserving successful execution state:"
        )

        print(
            json.dumps(
                execution["state"],
                indent=2
            )
        )

        # ----------------------------------------------------
        # Ask LLM for a revised plan.
        # ----------------------------------------------------

        recovery_plan = create_recovery_plan(
            plan,
            execution["failed_step"],
            execution["error"],
            execution["state"]
        )

        # ----------------------------------------------------
        # Validate recovery plan.
        # ----------------------------------------------------

        valid, message = validate_plan(
            recovery_plan
        )

        if not valid:

            print(
                f"\nRECOVERY PLAN REJECTED: "
                f"{message}"
            )

            return

        # ----------------------------------------------------
        # Execute recovery plan using the existing state.
        # ----------------------------------------------------

        execution = execute_plan(
            recovery_plan,
            "C123",
            existing_state=execution["state"]
        )

    # ========================================================
    # PHASE 5 — FINAL STATE VALIDATION
    # ========================================================

    final_state_valid = (
        validate_final_state(
            execution["state"]
        )
    )

    # ========================================================
    # PHASE 6 — FINAL RESULT
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL STATE"
    )

    print(
        "=" * 70
    )

    print(
        json.dumps(
            execution["state"],
            indent=2
        )
    )

    print(
        "\n" + "=" * 70
    )

    if (
        execution["success"]
        and final_state_valid
    ):

        print(
            "WORKFLOW: COMPLETED"
        )

    else:

        print(
            "WORKFLOW: FAILED"
        )

    print(
        "=" * 70
    )


# ============================================================
# 16. APPLICATION ENTRY POINT
# ============================================================


if __name__ == "__main__":

    main()