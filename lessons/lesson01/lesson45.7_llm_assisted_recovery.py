"""
Lesson 45.7 — LLM-Assisted Recovery Planning

Goal:
    Demonstrate how an LLM can recommend a recovery strategy
    after a tool failure, while deterministic application logic
    validates the recommendation.

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
    Current State
         ↓
    LLM Recovery Recommendation
         ↓
    Deterministic Policy Validation
         ↓
    Recovery / Stop
         ↓
    Final LLM

Key principle:

    LLM recommends.
    Application policy validates.
    LangGraph orchestrates.
    Tools execute.

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

                # Some MCP servers return an error as text.
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

A workflow operation has failed.

User request:
{state["user_request"]}

Current workflow state:

Employee:
{state.get("employee")}

Employee ID:
{state.get("employee_id")}

Product requested:
{state.get("product_name")}

Product result:
{state.get("product")}

Product error:
{state.get("product_error")}

Choose the safest useful recovery strategy.

Allowed recommendations:

1. CONTINUE_PARTIALLY
   Use this when enough trustworthy information exists
   to answer part of the request.

2. FALLBACK
   Use this only when another capability should be used
   to obtain the missing information.

3. STOP
   Use this when the workflow cannot safely continue.

Return ONLY JSON:

{{
    "recommendation": "CONTINUE_PARTIALLY",
    "reason": "..."
}}

Do not invent missing information.
"""

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt
    )

    response = interaction.output_text

    print("RAW RECOVERY RECOMMENDATION:")
    print(response)

    response = clean_json_response(response)

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
# STEP 6 — DETERMINISTIC POLICY VALIDATION
# ============================================================

async def validate_recovery_node(state: AgentState):

    print("\n" + "=" * 60)
    print("STEP 6 — RECOVERY POLICY VALIDATION")
    print("=" * 60)

    recommendation = (
        state.get(
            "llm_recovery_recommendation"
        )
    )

    print(
        "LLM recommended:",
        recommendation
    )

    # --------------------------------------------------------
    # Application-controlled allowed actions
    # --------------------------------------------------------

    ALLOWED_RECOVERY_ACTIONS = {
        "CONTINUE_PARTIALLY",
        "FALLBACK",
        "STOP"
    }

    if recommendation not in ALLOWED_RECOVERY_ACTIONS:

        print(
            "POLICY: Invalid recovery recommendation"
        )

        return {
            "approved_recovery_action": "STOP",
            "policy_reason": (
                "LLM recommendation was not in the "
                "allowed recovery action set."
            )
        }

    # --------------------------------------------------------
    # Deterministic safety rule
    #
    # We know employee information is valid.
    # Therefore CONTINUE_PARTIALLY is allowed.
    #
    # We do NOT allow the LLM to invent a fallback tool.
    # --------------------------------------------------------

    if recommendation == "CONTINUE_PARTIALLY":

        if (
            state.get("employee")
            and state.get("employee_id") is not None
        ):

            approved_action = "CONTINUE_PARTIALLY"

            policy_reason = (
                "Employee information is valid and available. "
                "Partial continuation is permitted."
            )

        else:

            approved_action = "STOP"

            policy_reason = (
                "Required employee state is missing. "
                "Partial continuation is not permitted."
            )

    elif recommendation == "FALLBACK":

        # No fallback capability has been defined in this
        # experiment, so the application refuses it.

        approved_action = "STOP"

        policy_reason = (
            "LLM requested FALLBACK, but no approved fallback "
            "capability is available in this workflow."
        )

    else:

        approved_action = "STOP"

        policy_reason = (
            "Workflow will stop because the recovery planner "
            "recommended STOP."
        )

    print("\nPOLICY DECISION:")
    print(approved_action)

    print("\nPOLICY REASON:")
    print(policy_reason)

    return {
        "approved_recovery_action": approved_action,
        "policy_reason": policy_reason
    }


# ============================================================
# STEP 7 — RECOVERY EXECUTION
# ============================================================

async def recovery_execution_node(state: AgentState):

    print("\n" + "=" * 60)
    print("STEP 7 — RECOVERY EXECUTION")
    print("=" * 60)

    action = state.get(
        "approved_recovery_action"
    )

    if action == "CONTINUE_PARTIALLY":

        print(
            "Recovery action: Continue using successfully "
            "retrieved employee information."
        )

    elif action == "STOP":

        print(
            "Recovery action: Stop safely."
        )

    else:

        print(
            "Unexpected recovery action."
        )

    return {}


# ============================================================
# STEP 8 — FINAL LLM
# ============================================================

async def final_node(state: AgentState):

    print("\n" + "=" * 60)
    print("STEP 8 — FINAL LLM")
    print("=" * 60)

    prompt = f"""
Answer the user's request using ONLY the workflow state.

User request:
{state["user_request"]}

Employee:
{state.get("employee")}

Product requested:
{state.get("product_name")}

Product result:
{state.get("product")}

Product error:
{state.get("product_error")}

LLM recovery recommendation:
{state.get("llm_recovery_recommendation")}

LLM recovery reason:
{state.get("llm_recovery_reason")}

Approved application recovery action:
{state.get("approved_recovery_action")}

Application policy reason:
{state.get("policy_reason")}

Important:

- Do not invent information.
- Employee information was successfully retrieved.
- Product lookup failed.
- Do not claim product availability.
- Clearly distinguish retrieved information from unavailable
  information.
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
    "recovery_execution",
    recovery_execution_node
)

graph.add_node(
    "final",
    final_node
)


# ============================================================
# WORKFLOW
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

# Product failure is handled by the recovery planner.
graph.add_edge(
    "product",
    "recovery_planner"
)

# LLM recommendation goes through deterministic validation.
graph.add_edge(
    "recovery_planner",
    "validate_recovery"
)

graph.add_edge(
    "validate_recovery",
    "recovery_execution"
)

graph.add_edge(
    "recovery_execution",
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