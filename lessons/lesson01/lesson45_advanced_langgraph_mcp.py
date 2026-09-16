"""
Lesson 45 — Advanced LangGraph + MCP Patterns

Goal:
Combine LLM + LangGraph + shared state + an existing MCP server.

Existing MCP server reused:
    lessons/lesson01/lesson13_mcp_server.py

MCP capabilities:
    find_employee
    get_employee
    get_salary

Architecture:

User Request
     ↓
LLM
     ↓
Structured Intent
     ↓
LangGraph
     ↓
MCP Client
     ↓
Existing Lesson 13 MCP Server
     ↓
MCP Tools
     ↓
LangGraph State
     ↓
Final LLM
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

    employee: str
    salary: str

    final_answer: str


# ============================================================
# 3. MCP SERVER CONFIGURATION
# ============================================================

server_params = StdioServerParameters(
    command="python",
    args=[
        "lessons/lesson01/lesson13_mcp_server.py"
    ],
)


# ============================================================
# 4. LLM — INTERPRET USER REQUEST
# ============================================================

def interpret_request(user_request: str) -> dict:

    prompt = f"""
You are an enterprise AI assistant.

Interpret the following request.

Return ONLY valid JSON.

User request:
{user_request}

Return exactly:

{{
    "employee_name": "Raj",
    "employee_id": 1
}}

Rules:

- If the employee ID is explicitly present, return it.
- If the employee name is present, return the name.
- If the request contains both, return both.
- Do not invent an employee.
- Return null when a value is not available.
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
# 5. LANGGRAPH — INTERPRETATION NODE
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
        "employee_name": result.get("employee_name") or "",
        "employee_id": result.get("employee_id")
    }


# ============================================================
# 6. MCP — FIND EMPLOYEE
# ============================================================

async def find_employee_mcp(
    session: ClientSession,
    name: str
) -> str:

    print("\n[MCP CLIENT] Calling find_employee")

    result = await session.call_tool(
        "find_employee",
        arguments={
            "name": name
        }
    )

    return result.content[0].text


# ============================================================
# 7. MCP — GET EMPLOYEE
# ============================================================

async def get_employee_mcp(
    session: ClientSession,
    employee_id: int
) -> str:

    print("\n[MCP CLIENT] Calling get_employee")

    result = await session.call_tool(
        "get_employee",
        arguments={
            "employee_id": employee_id
        }
    )

    return result.content[0].text


# ============================================================
# 8. MCP — GET SALARY
# ============================================================

async def get_salary_mcp(
    session: ClientSession,
    employee_id: int
) -> str:

    print("\n[MCP CLIENT] Calling get_salary")

    result = await session.call_tool(
        "get_salary",
        arguments={
            "employee_id": employee_id
        }
    )

    return result.content[0].text


# ============================================================
# 9. LANGGRAPH — EMPLOYEE INFORMATION NODE
# ============================================================

async def employee_node(state: AgentState):

    print("\n" + "=" * 60)
    print("STEP 2 — EMPLOYEE INFORMATION THROUGH MCP")
    print("=" * 60)

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()

            # Discover available MCP tools
            tools = await session.list_tools()

            print("\n[MCP] AVAILABLE TOOLS:")

            for tool in tools.tools:
                print(f"- {tool.name}")

            employee_id = state["employee_id"]

            # If we don't have ID, find employee by name
            if employee_id is None:

                employee_result = await find_employee_mcp(
                    session,
                    state["employee_name"]
                )

                print("\n[MCP RESULT]")
                print(employee_result)

                # For this lesson the database result is:
                # Employee ID: 1
                #
                # We extract the ID from the returned MCP text.

                for line in employee_result.splitlines():

                    if line.startswith("Employee ID:"):
                        employee_id = int(
                            line.split(":")[1].strip()
                        )

            if employee_id is None:

                return {
                    "employee": "Employee could not be identified."
                }

            employee_result = await get_employee_mcp(
                session,
                employee_id
            )

            print("\n[MCP RESULT]")
            print(employee_result)

            return {
                "employee_id": employee_id,
                "employee": employee_result
            }


# ============================================================
# 10. LANGGRAPH — SALARY NODE
# ============================================================

async def salary_node(state: AgentState):

    print("\n" + "=" * 60)
    print("STEP 3 — SALARY THROUGH MCP")
    print("=" * 60)

    employee_id = state["employee_id"]

    if employee_id is None:

        return {
            "salary": "Salary unavailable because employee ID is unknown."
        }

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()

            salary_result = await get_salary_mcp(
                session,
                employee_id
            )

            print("\n[MCP RESULT]")
            print(salary_result)

            return {
                "salary": salary_result
            }


# ============================================================
# 11. FINAL LLM
# ============================================================

def generate_final_answer(state: AgentState):

    print("\n" + "=" * 60)
    print("STEP 4 — FINAL LLM")
    print("=" * 60)

    prompt = f"""
You are an enterprise business assistant.

Answer the user's request using only the information
retrieved from the MCP tools.

User request:
{state["user_request"]}

Employee information:
{state["employee"]}

Salary information:
{state["salary"]}

Give a concise business-oriented answer.
Do not invent information.
"""

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt
    )

    return {
        "final_answer": interaction.output_text.strip()
    }


# ============================================================
# 12. BUILD LANGGRAPH
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
    "salary",
    salary_node
)

graph.add_node(
    "final_answer",
    generate_final_answer
)


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
    "salary"
)

graph.add_edge(
    "salary",
    "final_answer"
)

graph.add_edge(
    "final_answer",
    END
)


app = graph.compile()


# ============================================================
# 13. INITIAL STATE
# ============================================================

initial_state: AgentState = {

    "user_request":
        "Give me the details and salary of Raj.",

    "employee_name": "",

    "employee_id": None,

    "employee": "",

    "salary": "",

    "final_answer": ""
}


# ============================================================
# 14. RUN LANGGRAPH
# ============================================================

async def main():

    print("=" * 60)
    print("LESSON 45 — LANGGRAPH + EXISTING MCP")
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

    print("\nLESSON 45 STEP COMPLETE")


# ============================================================
# 15. START
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())