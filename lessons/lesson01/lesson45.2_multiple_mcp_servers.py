"""
Lesson 45.2 — Multiple MCP Servers + LangGraph

Goal:
Demonstrate one LangGraph workflow using multiple independent
MCP servers.

Existing MCP servers:

1. Employee / Finance MCP
   lessons/lesson01/lesson13_mcp_server.py

   Tools:
       find_employee
       get_employee
       get_salary

2. Product MCP
   lessons/lesson01/lesson12_mcp.py

   Tool:
       search_product

Architecture:

User Request
     ↓
LLM
     ↓
Intent
     ↓
LangGraph
     ├───────────────┐
     ↓               ↓
Employee MCP     Product MCP
     ↓               ↓
HR / Finance     Product System
     └───────────────┘
             ↓
        Shared State
             ↓
          Final LLM

Key concept:
Multiple independent MCP servers can participate in the
same LangGraph workflow.
"""


import os
import json
import asyncio

from typing import TypedDict

from dotenv import load_dotenv
from google import genai

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. SETUP
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found.")

client = genai.Client(api_key=api_key)


# ============================================================
# 2. SHARED LANGGRAPH STATE
# ============================================================

class AgentState(TypedDict):

    user_request: str

    employee_name: str
    employee_id: int | None
    product_name: str

    needs_employee: bool
    needs_product: bool

    employee: str
    product: str

    final_answer: str


# ============================================================
# 3. MCP SERVER CONFIGURATION
# ============================================================

# Employee / Finance MCP server

employee_server_params = StdioServerParameters(
    command="python",
    args=[
        "lessons/lesson01/lesson13_mcp_server.py"
    ],
)


# Product MCP server

product_server_params = StdioServerParameters(
    command="python",
    args=[
        "lessons/lesson01/lesson12_mcp.py"
    ],
)


# ============================================================
# 4. LLM — INTERPRET USER REQUEST
# ============================================================

def interpret_request(user_request: str) -> dict:

    prompt = f"""
You are an enterprise AI assistant.

Analyze the user's request.

Return ONLY valid JSON.

User request:
{user_request}

Return exactly:

{{
    "employee_name": "Raj",
    "employee_id": 1,
    "product_name": "MacBook Air",
    "needs_employee": true,
    "needs_product": true
}}

Rules:

1. employee_name:
   Return the employee name if present.
   Otherwise return null.

2. employee_id:
   Return the employee ID if explicitly present.
   Otherwise return null.

3. product_name:
   Return the product name if a product is mentioned.
   Otherwise return null.

4. needs_employee:
   true if employee information is requested.

5. needs_product:
   true if product information is requested.

6. Do not invent information.
"""

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt
    )

    text = interaction.output_text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

    return json.loads(text)


# ============================================================
# 5. INTERPRETATION NODE
# ============================================================

def interpret_node(state: AgentState):

    print("\n" + "=" * 60)
    print("STEP 1 — LLM INTERPRETATION")
    print("=" * 60)

    result = interpret_request(
        state["user_request"]
    )

    print(json.dumps(result, indent=2))

    return {
        "employee_name":
            result.get("employee_name") or "",

        "employee_id":
            result.get("employee_id"),

        "product_name":
            result.get("product_name") or "",

        "needs_employee":
            result.get("needs_employee", False),

        "needs_product":
            result.get("needs_product", False)
    }


# ============================================================
# 6. EMPLOYEE MCP NODE
# ============================================================

async def employee_node(state: AgentState):

    print("\n" + "=" * 60)
    print("STEP 2 — EMPLOYEE MCP SERVER")
    print("=" * 60)

    async with stdio_client(
        employee_server_params
    ) as (read, write):

        async with ClientSession(
            read,
            write
        ) as session:

            await session.initialize()

            # Discover tools

            tools = await session.list_tools()

            print("\nAVAILABLE EMPLOYEE MCP TOOLS:")

            for tool in tools.tools:
                print(f"- {tool.name}")

            employee_id = state["employee_id"]

            # Find employee if ID isn't known

            if employee_id is None:

                result = await session.call_tool(
                    "find_employee",
                    arguments={
                        "name":
                            state["employee_name"]
                    }
                )

                text = result.content[0].text

                print("\nfind_employee RESULT:")
                print(text)

                for line in text.splitlines():

                    if line.startswith("Employee ID:"):

                        employee_id = int(
                            line.split(":")[1].strip()
                        )

            if employee_id is None:

                return {
                    "employee":
                        "Employee could not be identified."
                }

            # Get employee information

            result = await session.call_tool(
                "get_employee",
                arguments={
                    "employee_id":
                        employee_id
                }
            )

            employee_result = result.content[0].text

            print("\nget_employee RESULT:")
            print(employee_result)

            return {
                "employee_id":
                    employee_id,

                "employee":
                    employee_result
            }


# ============================================================
# 7. PRODUCT MCP NODE
# ============================================================

async def product_node(state: AgentState):

    print("\n" + "=" * 60)
    print("STEP 3 — PRODUCT MCP SERVER")
    print("=" * 60)

    async with stdio_client(
        product_server_params
    ) as (read, write):

        async with ClientSession(
            read,
            write
        ) as session:

            await session.initialize()

            # Discover tools

            tools = await session.list_tools()

            print("\nAVAILABLE PRODUCT MCP TOOLS:")

            for tool in tools.tools:
                print(f"- {tool.name}")

            result = await session.call_tool(
                "search_product",
                arguments={
                    "product_name":
                        state["product_name"]
                }
            )

            product_result = result.content[0].text

            print("\nsearch_product RESULT:")
            print(product_result)

            return {
                "product":
                    product_result
            }


# ============================================================
# 8. FINAL LLM
# ============================================================

def final_llm_node(state: AgentState):

    print("\n" + "=" * 60)
    print("STEP 4 — FINAL LLM")
    print("=" * 60)

    prompt = f"""
You are an enterprise business assistant.

Answer the user's request using the retrieved information.

User request:
{state["user_request"]}

Employee information:
{state["employee"]}

Product information:
{state["product"]}

Do not invent information.

Give a concise business answer.
"""

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt
    )

    return {
        "final_answer":
            interaction.output_text.strip()
    }


# ============================================================
# 9. BUILD LANGGRAPH
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
    "product",
    product_node
)

graph.add_node(
    "final",
    final_llm_node
)


# ============================================================
# 10. WORKFLOW
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
    "product"
)

graph.add_edge(
    "product",
    "final"
)

graph.add_edge(
    "final",
    END
)


app = graph.compile()


# ============================================================
# 11. INITIAL STATE
# ============================================================

initial_state: AgentState = {

    "user_request":
        "Give me Raj's department and tell me whether "
        "MacBook Air is available.",

    "employee_name": "",

    "employee_id": None,

    "product_name": "",

    "needs_employee": False,

    "needs_product": False,

    "employee": "",

    "product": "",

    "final_answer": ""
}


# ============================================================
# 12. RUN
# ============================================================

async def main():

    print("=" * 60)
    print("LESSON 45.2 — MULTIPLE MCP SERVERS")
    print("=" * 60)

    result = await app.ainvoke(
        initial_state
    )

    print("\n" + "=" * 60)
    print("FINAL BUSINESS ANSWER")
    print("=" * 60)

    print(result["final_answer"])

    print("\n" + "=" * 60)
    print("FINAL LANGGRAPH STATE")
    print("=" * 60)

    print(
        json.dumps(
            result,
            indent=2
        )
    )

    print("\nLESSON 45.2 COMPLETE")


if __name__ == "__main__":
    asyncio.run(main())