"""
Lesson 45.4 — Dependency-Aware MCP Orchestration

Goal:
    Demonstrate how LangGraph handles MCP operations where
    one operation depends on the result of another.

Architecture:

    User Request
          ↓
        LLM
          ↓
    Employee MCP
          ↓
    Employee Result
          ↓
     Product MCP
          ↓
      Final LLM

Key principle:
    Independent work can run in parallel.
    Dependent work must run sequentially.

Runtime:
    GitHub Codespaces
    Gemini Interactions API
    LangGraph
    MCP Python SDK

MCP servers reused:
    Lesson 13 Employee MCP
    Lesson 12 Product MCP
"""
from typing import TypedDict, Optional

import asyncio
import json

from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
from langgraph.graph import StateGraph, START, END
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ============================================================
# LLM
# ============================================================

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


# ============================================================
# STATE
# ============================================================

class AgentState(TypedDict, total=False):
    user_request: str
    employee_name: Optional[str]
    employee_id: Optional[int]
    product_name: Optional[str]
    employee: Optional[str]
    product: Optional[str]
    final_answer: Optional[str]

# ============================================================
# STEP 1 — LLM INTERPRETATION
# ============================================================

async def interpret_node(state):

    prompt = f"""
Interpret this business request.

Request:
{state["user_request"]}

Return ONLY valid JSON:

{{
    "employee_name": "...",
    "product_name": "..."
}}

The product name should be selected based on the employee information
that will later be retrieved from the employee MCP server.
If the product cannot yet be determined, return null.
"""

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt
    )

    response = interaction.output_text

    print("\nSTEP 1 — LLM INTERPRETATION")
    print(response)

    data = json.loads(response)

    return {
        "employee_name": data.get("employee_name"),
        "product_name": data.get("product_name")
    }


# ============================================================
# STEP 2 — EMPLOYEE MCP
# ============================================================

async def employee_node(state):

    print("\n[DEPENDENCY STEP 1] EMPLOYEE MCP STARTED")

    server_params = StdioServerParameters(
        command="python",
        args=["lessons/lesson01/lesson13_mcp_server.py"]
    )

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()

            tools_result = await session.list_tools()

            print(
                "Employee MCP tools:",
                [tool.name for tool in tools_result.tools]
            )

            employee_name = state["employee_name"]

            result = await session.call_tool(
                "find_employee",
                arguments={
                    "name": employee_name
                }
            )

            employee_result = result.content[0].text

            print("\nEMPLOYEE MCP RESULT:")
            print(employee_result)

            # Extract employee ID from the MCP response
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

    print("[DEPENDENCY STEP 1] EMPLOYEE MCP COMPLETED")

    return {
        "employee_id": employee_id,
        "employee": employee
    }


# ============================================================
# STEP 3 — DETERMINE PRODUCT FROM EMPLOYEE RESULT
# ============================================================

async def determine_product_node(state):

    print("\n[DEPENDENCY STEP 2] DETERMINING PRODUCT")

    prompt = f"""
Based on the employee information below, determine which product
should be checked.

Employee:
{state["employee"]}

Return ONLY valid JSON:

{{
    "product_name": "..."
}}
"""

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt
    )

    response = interaction.output_text

    print("PRODUCT DECISION:")
    print(response)

    data = json.loads(response)

    return {
        "product_name": data["product_name"]
    }


# ============================================================
# STEP 4 — PRODUCT MCP
# ============================================================

async def product_node(state):

    print("\n[DEPENDENCY STEP 3] PRODUCT MCP STARTED")

    server_params = StdioServerParameters(
        command="python",
        args=["lessons/lesson01/lesson12_mcp.py"]
    )

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()

            tools_result = await session.list_tools()

            print(
                "Product MCP tools:",
                [tool.name for tool in tools_result.tools]
            )

            result = await session.call_tool(
                "search_product",
                arguments={
                    "product_name": state["product_name"]
                }
            )

            product_result = result.content[0].text

            print("\nPRODUCT MCP RESULT:")
            print(product_result)

    print("[DEPENDENCY STEP 3] PRODUCT MCP COMPLETED")

    return {
        "product": product_result
    }


# ============================================================
# STEP 5 — FINAL LLM
# ============================================================

async def final_node(state):

    print("\nSTEP 5 — FINAL LLM")

    prompt = f"""
Answer the user's request using ONLY the information available
in the state.

User request:
{state["user_request"]}

Employee:
{state.get("employee")}

Product:
{state.get("product")}

Do not invent product availability or other facts that are not
explicitly present in the tool result.
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

graph.add_node("interpret", interpret_node)
graph.add_node("employee", employee_node)
graph.add_node("determine_product", determine_product_node)
graph.add_node("product", product_node)
graph.add_node("final", final_node)


# ------------------------------------------------------------
# Dependency structure
# ------------------------------------------------------------

graph.add_edge(START, "interpret")

graph.add_edge("interpret", "employee")

# Product decision MUST wait for employee result
graph.add_edge("employee", "determine_product")

# Product MCP MUST wait for product decision
graph.add_edge("determine_product", "product")

graph.add_edge("product", "final")

graph.add_edge("final", END)


app = graph.compile()


# ============================================================
# RUN
# ============================================================

async def main():

    user_request = (
        "Find Raj's department and then determine which product "
        "should be checked based on his employee information."
    )

    result = await app.ainvoke({
        "user_request": user_request
    })

    print("\n" + "=" * 60)
    print("FINAL STATE")
    print("=" * 60)

    print(result)


if __name__ == "__main__":
    asyncio.run(main())