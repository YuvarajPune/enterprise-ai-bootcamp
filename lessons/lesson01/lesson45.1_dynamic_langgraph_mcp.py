"""
Lesson 45.1 — Dynamic LangGraph Routing + MCP

Goal:
Demonstrate dynamic routing in LangGraph using an existing MCP server.

Existing MCP server:
    lessons/lesson01/lesson13_mcp_server.py

MCP tools:
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
LangGraph Router
     ↓
 ┌───────────────┬───────────────┐
 ↓               ↓               ↓
Employee       Salary           Both
MCP             MCP              MCP
 ↓               ↓               ↓
 └───────────────┴───────────────┘
                 ↓
             Final LLM
                 ↓
          Business Answer

Key concept:
The graph does not always execute every node.
The router selects the required execution path.
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
# 2. SHARED STATE
# ============================================================

class AgentState(TypedDict):

    user_request: str

    employee_name: str
    employee_id: int | None

    needs_employee: bool
    needs_salary: bool

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
# 4. LLM — INTERPRET REQUEST
# ============================================================

def interpret_request(user_request: str) -> dict:

    prompt = f"""
You are an enterprise AI assistant.

Analyze the user's request and return ONLY valid JSON.

User request:
{user_request}

Return exactly:

{{
    "employee_name": "Raj",
    "employee_id": 1,
    "needs_employee": true,
    "needs_salary": false
}}

Rules:

1. employee_name:
   Return the employee name if provided.
   Otherwise return null.

2. employee_id:
   Return the employee ID if explicitly provided.
   Otherwise return null.

3. needs_employee:
   true if the user needs employee information such as:
   name, department, employee ID or employee details.

4. needs_salary:
   true if the user asks for salary or compensation.

5. Do not invent information.
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
        "employee_name": result.get("employee_name") or "",
        "employee_id": result.get("employee_id"),
        "needs_employee": result.get("needs_employee", False),
        "needs_salary": result.get("needs_salary", False)
    }


# ============================================================
# 6. DYNAMIC ROUTER
# ============================================================

def route_request(state: AgentState):

    print("\n" + "=" * 60)
    print("STEP 2 — LANGGRAPH ROUTER")
    print("=" * 60)

    needs_employee = state["needs_employee"]
    needs_salary = state["needs_salary"]

    if needs_employee and needs_salary:

        print("ROUTE: BOTH")

        return "both"

    if needs_employee:

        print("ROUTE: EMPLOYEE")

        return "employee"

    if needs_salary:

        print("ROUTE: SALARY")

        return "salary"

    print("ROUTE: NONE")

    return "none"


# ============================================================
# 7. EMPLOYEE MCP NODE
# ============================================================

async def employee_node(state: AgentState):

    print("\n" + "=" * 60)
    print("EMPLOYEE NODE — MCP")
    print("=" * 60)

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()

            employee_id = state["employee_id"]

            # If ID is unknown, find employee by name
            if employee_id is None:

                result = await session.call_tool(
                    "find_employee",
                    arguments={
                        "name": state["employee_name"]
                    }
                )

                employee_result = result.content[0].text

                print("\nfind_employee RESULT:")
                print(employee_result)

                for line in employee_result.splitlines():

                    if line.startswith("Employee ID:"):

                        employee_id = int(
                            line.split(":")[1].strip()
                        )

            if employee_id is None:

                return {
                    "employee": "Employee could not be identified."
                }

            result = await session.call_tool(
                "get_employee",
                arguments={
                    "employee_id": employee_id
                }
            )

            employee_result = result.content[0].text

            print("\nget_employee RESULT:")
            print(employee_result)

            return {
                "employee_id": employee_id,
                "employee": employee_result
            }


# ============================================================
# 8. SALARY MCP NODE
# ============================================================

async def salary_node(state: AgentState):

    print("\n" + "=" * 60)
    print("SALARY NODE — MCP")
    print("=" * 60)

    employee_id = state["employee_id"]

    # If employee ID is already known
    if employee_id is None:

        print("Salary requires employee identification.")

        return {
            "salary":
                "Salary could not be retrieved because "
                "employee ID is unknown."
        }

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()

            result = await session.call_tool(
                "get_salary",
                arguments={
                    "employee_id": employee_id
                }
            )

            salary_result = result.content[0].text

            print("\nget_salary RESULT:")
            print(salary_result)

            return {
                "salary": salary_result
            }


# ============================================================
# 9. BOTH NODE
# ============================================================

async def both_node(state: AgentState):

    print("\n" + "=" * 60)
    print("BOTH NODE — MCP")
    print("=" * 60)

    employee_id = state["employee_id"]

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()

            # ----------------------------------------
            # Find employee if required
            # ----------------------------------------

            if employee_id is None:

                result = await session.call_tool(
                    "find_employee",
                    arguments={
                        "name": state["employee_name"]
                    }
                )

                employee_result = result.content[0].text

                print("\nfind_employee RESULT:")
                print(employee_result)

                for line in employee_result.splitlines():

                    if line.startswith("Employee ID:"):

                        employee_id = int(
                            line.split(":")[1].strip()
                        )

            if employee_id is None:

                return {
                    "employee":
                        "Employee could not be identified.",
                    "salary":
                        "Salary could not be retrieved."
                }

            # ----------------------------------------
            # Get employee information
            # ----------------------------------------

            employee_result = await session.call_tool(
                "get_employee",
                arguments={
                    "employee_id": employee_id
                }
            )

            employee_text = employee_result.content[0].text

            print("\nget_employee RESULT:")
            print(employee_text)

            # ----------------------------------------
            # Get salary
            # ----------------------------------------

            salary_result = await session.call_tool(
                "get_salary",
                arguments={
                    "employee_id": employee_id
                }
            )

            salary_text = salary_result.content[0].text

            print("\nget_salary RESULT:")
            print(salary_text)

            return {
                "employee_id": employee_id,
                "employee": employee_text,
                "salary": salary_text
            }


# ============================================================
# 10. NO ACTION NODE
# ============================================================

def no_action_node(state: AgentState):

    print("\n" + "=" * 60)
    print("NO INFORMATION REQUIRED")
    print("=" * 60)

    return {
        "employee":
            "No employee information requested.",
        "salary":
            "No salary information requested."
    }


# ============================================================
# 11. FINAL LLM
# ============================================================

def final_llm_node(state: AgentState):

    print("\n" + "=" * 60)
    print("FINAL LLM")
    print("=" * 60)

    prompt = f"""
You are an enterprise business assistant.

Answer the user's request using the retrieved information.

User request:
{state["user_request"]}

Employee information:
{state["employee"]}

Salary information:
{state["salary"]}

Do not invent information.
If requested information is unavailable, say so clearly.

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
# 12. BUILD LANGGRAPH
# ============================================================

graph = StateGraph(AgentState)


# Register nodes

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
    "both",
    both_node
)

graph.add_node(
    "none",
    no_action_node
)

graph.add_node(
    "final",
    final_llm_node
)


# ============================================================
# 13. DEFINE DYNAMIC ROUTING
# ============================================================

graph.add_edge(
    START,
    "interpret"
)


graph.add_conditional_edges(
    "interpret",
    route_request,
    {
        "employee": "employee",
        "salary": "salary",
        "both": "both",
        "none": "none"
    }
)


# All paths eventually reach final LLM

graph.add_edge(
    "employee",
    "final"
)

graph.add_edge(
    "salary",
    "final"
)

graph.add_edge(
    "both",
    "final"
)

graph.add_edge(
    "none",
    "final"
)

graph.add_edge(
    "final",
    END
)


# Compile

app = graph.compile()


# ============================================================
# 14. TEST REQUEST
# ============================================================

initial_state: AgentState = {

    "user_request":
        "What is Raj's department?",

    "employee_name": "",

    "employee_id": None,

    "needs_employee": False,

    "needs_salary": False,

    "employee": "",

    "salary": "",

    "final_answer": ""
}


# ============================================================
# 15. RUN
# ============================================================

async def main():

    print("=" * 60)
    print("LESSON 45.1 — DYNAMIC LANGGRAPH + MCP")
    print("=" * 60)

    result = await app.ainvoke(
        initial_state
    )

    print("\n" + "=" * 60)
    print("FINAL BUSINESS ANSWER")
    print("=" * 60)

    print(result["final_answer"])

    print("\n" + "=" * 60)
    print("FINAL STATE")
    print("=" * 60)

    print(
        json.dumps(
            result,
            indent=2
        )
    )

    print("\nLESSON 45.1 COMPLETE")


if __name__ == "__main__":
    asyncio.run(main())