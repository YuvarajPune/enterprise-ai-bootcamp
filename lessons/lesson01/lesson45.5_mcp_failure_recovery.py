"""
Lesson 45.5 — MCP Failure Recovery

Goal:
    Demonstrate failure recovery inside a dependency-aware
    LangGraph + MCP workflow.

Architecture:

    User Request
         ↓
      LLM Interpret
         ↓
    Employee MCP
         ↓
    LLM Determine Product
         ↓
    Product MCP
         ↓
       Failure
         ↓
      Router
         ↓
      Recovery
         ↓
      Final LLM

Key principle:
    A downstream tool failure does not necessarily mean
    the entire workflow must fail.

Existing MCP servers:
    Lesson 13 Employee MCP
    Lesson 12 Product MCP

Important:
    The Product MCP failure is intentional.
    We call a non-existent tool so that the recovery
    path can be demonstrated without modifying the
    existing MCP server.
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
        "Make sure your .env file contains the API key."
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

    recovery_action: Optional[str]

    final_answer: Optional[str]


# ============================================================
# HELPER — CLEAN LLM JSON
# ============================================================

def clean_json_response(response: str) -> str:
    """
    LLMs sometimes return JSON inside Markdown code fences.

    Example:

        ```json
        {
            "product_name": "Laptop"
        }
        ```

    This function removes the fences before json.loads().
    """

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
# STEP 1 — INTERPRET USER REQUEST
# ============================================================

async def interpret_node(state: AgentState):

    prompt = f"""
Interpret this business request.

Request:
{state["user_request"]}

Return ONLY JSON.

Format:

{{
    "employee_name": "Raj"
}}

Do not include Markdown or explanations.
"""

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt
    )

    response = interaction.output_text

    print("\n" + "=" * 60)
    print("STEP 1 — LLM INTERPRETATION")
    print("=" * 60)

    print(response)

    response = clean_json_response(response)

    try:

        data = json.loads(response)

    except json.JSONDecodeError as e:

        print("\nLLM JSON VALIDATION FAILED")
        print("ERROR:", e)

        raise ValueError(
            "LLM returned invalid JSON during request interpretation."
        )

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

            # ------------------------------------------------
            # Extract employee ID
            # ------------------------------------------------

            employee_id = None

            for line in employee_result.splitlines():

                if line.startswith("Employee ID:"):

                    employee_id = int(
                        line.split(":")[1].strip()
                    )

            # ------------------------------------------------
            # Get complete employee record
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
Based on the employee information below, determine which
product should be checked.

Employee information:

{state["employee"]}

Return ONLY JSON.

Format:

{{
    "product_name": "Laptop"
}}

Do not include Markdown or explanations.
"""

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt
    )

    response = interaction.output_text

    print("RAW PRODUCT DECISION:")
    print(response)

    # --------------------------------------------------------
    # Robustly handle Markdown JSON
    # --------------------------------------------------------

    response = clean_json_response(response)

    try:

        data = json.loads(response)

    except json.JSONDecodeError as e:

        print("\nLLM JSON VALIDATION FAILED")
        print("ERROR:", e)

        raise ValueError(
            "LLM returned invalid JSON while determining product."
        )

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

                # ------------------------------------------------
                # DELIBERATE FAILURE
                #
                # The real tool is:
                #
                #     search_product
                #
                # We intentionally call:
                #
                #     search_product_unavailable
                # ------------------------------------------------

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

                # Some MCP servers may return an error as normal text.
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
# STEP 5 — ROUTER
# ============================================================

def route_product_result(state: AgentState):

    print("\n" + "=" * 60)
    print("ROUTER")
    print("=" * 60)

    if state.get("product_error"):

        print("Product MCP failed.")
        print("ROUTE → RECOVERY")

        return "recovery"

    print("Product MCP succeeded.")
    print("ROUTE → FINAL")

    return "final"


# ============================================================
# STEP 6 — RECOVERY
# ============================================================

async def recovery_node(state: AgentState):

    print("\n" + "=" * 60)
    print("RECOVERY NODE")
    print("=" * 60)

    print(
        "Product lookup failed, but the employee information "
        "was retrieved successfully."
    )

    recovery_action = (
        "Continue with the successfully retrieved employee "
        "information and report that the product lookup "
        "was unavailable."
    )

    print("\nRECOVERY ACTION:")
    print(recovery_action)

    return {
        "recovery_action": recovery_action
    }


# ============================================================
# STEP 7 — FINAL LLM
# ============================================================

async def final_node(state: AgentState):

    print("\n" + "=" * 60)
    print("FINAL LLM")
    print("=" * 60)

    prompt = f"""
Answer the user's request using ONLY the information available
in the workflow state.

User request:
{state["user_request"]}

Employee:
{state.get("employee")}

Product:
{state.get("product")}

Product error:
{state.get("product_error")}

Recovery action:
{state.get("recovery_action")}

Important rules:

1. Employee information was successfully retrieved.
2. The product lookup failed.
3. Do NOT claim that the product was found.
4. Do NOT invent product availability.
5. Clearly distinguish successful information from unavailable
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
    "recovery",
    recovery_node
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


# Product result determines next path
graph.add_conditional_edges(
    "product",
    route_product_result,
    {
        "recovery": "recovery",
        "final": "final"
    }
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