"""
Lesson 45.8 — Human-in-the-Loop Recovery

Goal:
    Demonstrate human approval when an LLM recommends a
    higher-risk recovery action after an MCP failure.

Architecture:

    User Request
         ↓
      LLM
         ↓
    Employee MCP
         ↓
    LLM determines Product
         ↓
    Product MCP
         ↓
       FAILURE
         ↓
    LLM Recovery Planner
         ↓
    Recovery Policy
         ↓
    HIGH-RISK FALLBACK
         ↓
    HUMAN APPROVAL
       ↙       ↘
   APPROVE     REJECT
      ↓           ↓
   Fallback      Stop
      ↓
     Final LLM

Key principle:

    LLM recommends.
    Application policy determines whether approval is required.
    Human authorizes the high-risk recovery.
    Application executes the approved action.

Existing MCP servers:
    Lesson 13 Employee MCP
    Lesson 12 Product MCP

The Product MCP failure is intentional.
"""


import asyncio
import json
import os

from typing import TypedDict, Optional

from dotenv import load_dotenv
from google import genai

from langgraph.graph import StateGraph, START, END

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found. "
        "Make sure your .env file contains GEMINI_API_KEY."
    )

client = genai.Client(api_key=api_key)


# ============================================================
# STATE
# ============================================================

class AgentState(TypedDict, total=False):

    user_request: str

    employee_name: Optional[str]
    employee_id: Optional[int]
    employee: Optional[str]

    product_name: Optional[str]
    product: Optional[str]
    product_error: Optional[str]

    llm_recovery_recommendation: Optional[str]
    llm_recovery_reason: Optional[str]

    approved_recovery_action: Optional[str]
    policy_reason: Optional[str]

    human_approval_required: Optional[bool]
    human_decision: Optional[str]

    fallback_result: Optional[str]

    final_answer: Optional[str]


# ============================================================
# HELPER — CLEAN LLM JSON
# ============================================================

def clean_json_response(response: str) -> str:

    response = response.strip()

    if response.startswith("```"):

        lines = response.splitlines()

        cleaned_lines = []

        for line in lines:

            if line.strip().startswith("```"):
                continue

            cleaned_lines.append(line)

        response = "\n".join(cleaned_lines).strip()

    return response


# ============================================================
# STEP 1 — INTERPRET REQUEST
# ============================================================

async def interpret_node(state: AgentState):

    print("\n" + "=" * 60)
    print("STEP 1 — LLM INTERPRETATION")
    print("=" * 60)

    prompt = f"""
Interpret this business request.

Request:
{state["user_request"]}

Return ONLY JSON:

{{
    "employee_name": "Raj"
}}

Do not include Markdown or explanation.
"""

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt
    )

    response = clean_json_response(
        interaction.output_text
    )

    print(response)

    data = json.loads(response)

    return {
        "employee_name": data.get("employee_name")
    }


# ============================================================
# STEP 2 — EMPLOYEE MCP
# ============================================================

async def employee_node(state: AgentState):

    print("\n" + "=" * 60)
    print("EMPLOYEE MCP")
    print("=" * 60)

    print("EMPLOYEE MCP STARTED")

    server_params = StdioServerParameters(
        command="python",
        args=[
            "lessons/lesson01/lesson13_mcp_server.py"
        ]
    )

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()

            tools_result = await session.list_tools()

            print(
                "Employee MCP tools:",
                [tool.name for tool in tools_result.tools]
            )

            result = await session.call_tool(
                "find_employee",
                arguments={
                    "name": state["employee_name"]
                }
            )

            employee_result = result.content[0].text

            print("\nEMPLOYEE MCP RESULT:")
            print(employee_result)

            employee_id = None

            for line in employee_result.splitlines():

                if line.startswith("Employee ID:"):

                    employee_id = int(
                        line.split(":")[1].strip()
                    )

            employee = None

            if employee_id is not None:

                result = await session.call_tool(
                    "get_employee",
                    arguments={
                        "employee_id": employee_id
                    }
                )

                employee = result.content[0].text

                print("\nGET EMPLOYEE RESULT:")
                print(employee)

    print("\nEMPLOYEE MCP COMPLETED")

    return {
        "employee_id": employee_id,
        "employee": employee
    }


# ============================================================
# STEP 3 — DETERMINE PRODUCT
# ============================================================

async def determine_product_node(state: AgentState):

    print("\n" + "=" * 60)
    print("STEP 3 — DETERMINE PRODUCT")
    print("=" * 60)

    prompt = f"""
Based on this employee information, determine which product
should be checked.

Employee:
{state["employee"]}

Return ONLY JSON:

{{
    "product_name": "Laptop"
}}

Do not include Markdown or explanation.
"""

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt
    )

    response = clean_json_response(
        interaction.output_text
    )

    print("RAW PRODUCT DECISION:")
    print(response)

    data = json.loads(response)

    product_name = data.get("product_name")

    if not product_name:

        raise ValueError(
            "LLM did not provide a product_name."
        )

    print("\nVALIDATED PRODUCT:", product_name)

    return {
        "product_name": product_name
    }


# ============================================================
# STEP 4 — PRODUCT MCP
#
# INTENTIONAL FAILURE
# ============================================================

async def product_node(state: AgentState):

    print("\n" + "=" * 60)
    print("PRODUCT MCP")
    print("=" * 60)

    print("PRODUCT MCP STARTED")

    try:

        server_params = StdioServerParameters(
            command="python",
            args=[
                "lessons/lesson01/lesson12_mcp.py"
            ]
        )

        async with stdio_client(server_params) as (read, write):

            async with ClientSession(read, write) as session:

                await session.initialize()

                tools_result = await session.list_tools()

                print(
                    "Product MCP tools:",
                    [tool.name for tool in tools_result.tools]
                )

                print(
                    "\nCalling intentionally unavailable "
                    "tool to demonstrate failure..."
                )

                result = await session.call_tool(
                    "search_product_unavailable",
                    arguments={
                        "product_name": state["product_name"]
                    }
                )

                product_result = result.content[0].text

                print("\nPRODUCT MCP RESULT:")
                print(product_result)

                if getattr(result, "isError", False):

                    raise RuntimeError(product_result)

                if product_result.startswith("Unknown tool:"):

                    raise RuntimeError(product_result)

                return {
                    "product": product_result,
                    "product_error": None
                }

    except Exception as e:

        error_message = str(e)

        print("\nPRODUCT MCP FAILED")
        print("ERROR:", error_message)

        return {
            "product": None,
            "product_error": error_message
        }


# ============================================================
# STEP 5 — LLM RECOVERY PLANNER
# ============================================================

async def llm_recovery_planner_node(state: AgentState):

    print("\n" + "=" * 60)
    print("STEP 5 — LLM RECOVERY PLANNER")
    print("=" * 60)

    prompt = f"""
You are a recovery planning assistant.

A product lookup has failed.

User request:
{state["user_request"]}

Current state:

Employee:
{state.get("employee")}

Employee ID:
{state.get("employee_id")}

Product:
{state.get("product_name")}

Product error:
{state.get("product_error")}

Choose the safest recovery strategy.

Allowed recommendations:

1. CONTINUE_PARTIALLY
   Use this when enough trustworthy information exists
   to answer part of the request.

2. FALLBACK
   Use this when another approved capability should be
   used to obtain the missing information.

3. STOP
   Use this when the workflow cannot safely continue.

For this exercise, consider FALLBACK when a product lookup
failure prevents completion of the user's requested task.

Return ONLY JSON:

{{
    "recommendation": "FALLBACK",
    "reason": "..."
}}

Do not invent product information.
"""

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt
    )

    response = clean_json_response(
        interaction.output_text
    )

    print("RAW RECOVERY RECOMMENDATION:")
    print(response)

    data = json.loads(response)

    recommendation = data.get(
        "recommendation"
    )

    reason = data.get(
        "reason"
    )

    print("\nLLM RECOMMENDATION:")
    print(recommendation)

    print("\nLLM REASON:")
    print(reason)

    return {
        "llm_recovery_recommendation": recommendation,
        "llm_recovery_reason": reason
    }


# ============================================================
# STEP 6 — DETERMINISTIC RECOVERY POLICY
# ============================================================

async def validate_recovery_node(state: AgentState):

    print("\n" + "=" * 60)
    print("STEP 6 — RECOVERY POLICY")
    print("=" * 60)

    recommendation = state.get(
        "llm_recovery_recommendation"
    )

    print(
        "LLM recommended:",
        recommendation
    )

    ALLOWED_ACTIONS = {
        "CONTINUE_PARTIALLY",
        "FALLBACK",
        "STOP"
    }

    if recommendation not in ALLOWED_ACTIONS:

        print("Invalid LLM recommendation.")

        return {
            "approved_recovery_action": "STOP",
            "policy_reason": (
                "Invalid recovery recommendation."
            ),
            "human_approval_required": False
        }

    # --------------------------------------------------------
    # FALLBACK is considered higher risk in this exercise.
    # It therefore requires human approval.
    # --------------------------------------------------------

    if recommendation == "FALLBACK":

        approved_action = "FALLBACK"

        policy_reason = (
            "Fallback would invoke an alternative capability "
            "to complete the failed product lookup. "
            "This recovery action requires human approval."
        )

        human_approval_required = True

    elif recommendation == "CONTINUE_PARTIALLY":

        if state.get("employee"):

            approved_action = "CONTINUE_PARTIALLY"

            policy_reason = (
                "Employee information is valid. "
                "Partial continuation is allowed."
            )

            human_approval_required = False

        else:

            approved_action = "STOP"

            policy_reason = (
                "Required state is missing."
            )

            human_approval_required = False

    else:

        approved_action = "STOP"

        policy_reason = (
            "Recovery planner recommended stopping."
        )

        human_approval_required = False

    print("\nPOLICY DECISION:")
    print(approved_action)

    print("\nPOLICY REASON:")
    print(policy_reason)

    print(
        "\nHUMAN APPROVAL REQUIRED:",
        human_approval_required
    )

    return {
        "approved_recovery_action": approved_action,
        "policy_reason": policy_reason,
        "human_approval_required": human_approval_required
    }


# ============================================================
# STEP 7 — HUMAN APPROVAL
# ============================================================

async def human_approval_node(state: AgentState):

    print("\n" + "=" * 60)
    print("STEP 7 — HUMAN APPROVAL")
    print("=" * 60)

    print(
        "\nThe recovery action is considered high-risk."
    )

    print(
        "Requested recovery:",
        state.get("approved_recovery_action")
    )

    print(
        "Reason:",
        state.get("policy_reason")
    )

    print(
        "\nThe workflow is waiting for human authorization."
    )

    decision = input(
        "\nApprove fallback recovery? (yes/no): "
    ).strip().lower()

    if decision == "yes":

        human_decision = "APPROVED"

        print("\nHUMAN DECISION: APPROVED")

    else:

        human_decision = "REJECTED"

        print("\nHUMAN DECISION: REJECTED")

    return {
        "human_decision": human_decision
    }


# ============================================================
# STEP 8 — ROUTE AFTER HUMAN DECISION
# ============================================================

def route_after_human(state: AgentState):

    if state.get("human_decision") == "APPROVED":

        print("\nROUTER → FALLBACK EXECUTION")

        return "fallback"

    print("\nROUTER → FINAL")

    return "final"


# ============================================================
# STEP 9 — FALLBACK EXECUTION
# ============================================================

async def fallback_node(state: AgentState):

    print("\n" + "=" * 60)
    print("STEP 9 — FALLBACK EXECUTION")
    print("=" * 60)

    print(
        "Human approved the fallback."
    )

    print(
        "Executing approved fallback capability..."
    )

    # --------------------------------------------------------
    # Simulated fallback capability.
    #
    # We intentionally do NOT claim real product availability.
    # The fallback only demonstrates successful recovery
    # execution after approval.
    # --------------------------------------------------------

    fallback_result = (
        f"Fallback product lookup completed for "
        f"{state['product_name']}."
    )

    print("\nFALLBACK RESULT:")
    print(fallback_result)

    return {
        "fallback_result": fallback_result
    }


# ============================================================
# STEP 10 — FINAL LLM
# ============================================================

async def final_node(state: AgentState):

    print("\n" + "=" * 60)
    print("STEP 10 — FINAL LLM")
    print("=" * 60)

    prompt = f"""
Answer the user's request using ONLY the workflow state.

User request:
{state["user_request"]}

Employee:
{state.get("employee")}

Product requested:
{state.get("product_name")}

Original product result:
{state.get("product")}

Original product error:
{state.get("product_error")}

LLM recovery recommendation:
{state.get("llm_recovery_recommendation")}

Policy decision:
{state.get("approved_recovery_action")}

Human decision:
{state.get("human_decision")}

Fallback result:
{state.get("fallback_result")}

Important:

- Do not invent product availability.
- Clearly distinguish the original failed lookup from
  the fallback result.
- If human approval was rejected, say that the recovery
  was not executed.
- If fallback was executed, report only what the fallback
  result actually states.
"""

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt
    )

    answer = interaction.output_text

    print("\nFINAL BUSINESS ANSWER:")
    print(answer)

    return {
        "final_answer": answer
    }


# ============================================================
# BUILD LANGGRAPH
# ============================================================

graph = StateGraph(AgentState)


graph.add_node(
    "interpret",
    interpret_node
)

graph.add_node(
    "employee",
    employee_node
)

graph.add_node(
    "determine_product",
    determine_product_node
)

graph.add_node(
    "product",
    product_node
)

graph.add_node(
    "recovery_planner",
    llm_recovery_planner_node
)

graph.add_node(
    "validate_recovery",
    validate_recovery_node
)

graph.add_node(
    "human_approval",
    human_approval_node
)

graph.add_node(
    "fallback",
    fallback_node
)

graph.add_node(
    "final",
    final_node
)


# ============================================================
# WORKFLOW EDGES
# ============================================================

graph.add_edge(
    START,
    "interpret"
)

graph.add_edge(
    "interpret",
    "employee"
)

graph.add_edge(
    "employee",
    "determine_product"
)

graph.add_edge(
    "determine_product",
    "product"
)

graph.add_edge(
    "product",
    "recovery_planner"
)

graph.add_edge(
    "recovery_planner",
    "validate_recovery"
)


# ------------------------------------------------------------
# Policy determines whether human approval is needed.
# ------------------------------------------------------------

def route_policy(state: AgentState):

    if state.get("human_approval_required"):

        return "human"

    return "final"


graph.add_conditional_edges(
    "validate_recovery",
    route_policy,
    {
        "human": "human_approval",
        "final": "final"
    }
)


# ------------------------------------------------------------
# Human determines whether fallback executes.
# ------------------------------------------------------------

graph.add_conditional_edges(
    "human_approval",
    route_after_human,
    {
        "fallback": "fallback",
        "final": "final"
    }
)

graph.add_edge(
    "fallback",
    "final"
)

graph.add_edge(
    "final",
    END
)


# ============================================================
# COMPILE
# ============================================================

app = graph.compile()


# ============================================================
# MAIN
# ============================================================

async def main():

    user_request = (
        "Find Raj's department and then determine which "
        "product should be checked based on his employee "
        "information."
    )

    result = await app.ainvoke(
        {
            "user_request": user_request
        }
    )

    print("\n" + "=" * 60)
    print("FINAL STATE")
    print("=" * 60)

    print(result)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    asyncio.run(main())