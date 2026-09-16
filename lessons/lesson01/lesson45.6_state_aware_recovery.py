"""
Lesson 45.6 — State-Aware Recovery

Goal:
    Demonstrate how an agent can inspect the state it has already
    collected after a tool failure and decide whether it can:

        1. Continue partially
        2. Use a fallback
        3. Stop safely

Architecture:

    User Request
         ↓
      LLM
         ↓
    Employee MCP
         ↓
    Employee State
         ↓
    LLM determines Product
         ↓
    Product MCP
         ↓
       FAILURE
         ↓
    State-Aware Recovery
         ↓
    Final LLM

Key principle:

    Recovery should depend on what the agent already knows,
    not only on the fact that a tool failed.

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

    recovery_decision: Optional[str]
    recovery_reason: Optional[str]

    final_answer: Optional[str]


# ============================================================
# HELPER — CLEAN JSON
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

    response = interaction.output_text

    print(response)

    response = clean_json_response(response)

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

            # ------------------------------------------------
            # Find employee
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Get employee
            # ------------------------------------------------

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

    response = interaction.output_text

    print("RAW PRODUCT DECISION:")
    print(response)

    response = clean_json_response(response)

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

                # ------------------------------------------------
                # MCP may return an error as normal text.
                # ------------------------------------------------

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
# STEP 5 — STATE-AWARE RECOVERY
# ============================================================

async def state_aware_recovery_node(state: AgentState):

    print("\n" + "=" * 60)
    print("STATE-AWARE RECOVERY")
    print("=" * 60)

    print("\nInspecting current workflow state...")

    employee_available = bool(
        state.get("employee")
    )

    employee_id_available = (
        state.get("employee_id") is not None
    )

    product_available = bool(
        state.get("product")
    )

    product_error = state.get(
        "product_error"
    )

    print(
        "Employee information available:",
        employee_available
    )

    print(
        "Employee ID available:",
        employee_id_available
    )

    print(
        "Product information available:",
        product_available
    )

    print(
        "Product error:",
        product_error
    )

    # --------------------------------------------------------
    # Deterministic recovery policy
    # --------------------------------------------------------

    if employee_available and employee_id_available:

        decision = "CONTINUE_PARTIALLY"

        reason = (
            "Employee information was successfully retrieved. "
            "Product lookup failed, but the workflow can safely "
            "continue using the valid employee information."
        )

    elif product_available:

        decision = "CONTINUE"

        reason = (
            "Product information is available, so the workflow "
            "can continue."
        )

    else:

        decision = "STOP"

        reason = (
            "Required information is missing. "
            "The workflow cannot safely continue."
        )

    print("\nRECOVERY DECISION:")
    print(decision)

    print("\nRECOVERY REASON:")
    print(reason)

    return {
        "recovery_decision": decision,
        "recovery_reason": reason
    }


# ============================================================
# STEP 6 — FINAL LLM
# ============================================================

async def final_node(state: AgentState):

    print("\n" + "=" * 60)
    print("FINAL LLM")
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

Recovery decision:
{state.get("recovery_decision")}

Recovery reason:
{state.get("recovery_reason")}

Important rules:

1. Do not invent information.
2. Employee information was successfully retrieved.
3. The product lookup failed.
4. Do not claim the product was found or is available.
5. Clearly explain what was successfully retrieved and
   what could not be retrieved.
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
    "recovery",
    state_aware_recovery_node
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

# Product failure always goes to state-aware recovery
graph.add_edge(
    "product",
    "recovery"
)

graph.add_edge(
    "recovery",
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