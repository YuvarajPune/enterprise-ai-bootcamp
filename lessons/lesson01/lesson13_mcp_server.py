import asyncio
import sqlite3

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types


server = Server(
    "Employee Data Server",
    version="1.0"
)


# -----------------------------
# HR Database Tool
# -----------------------------

async def get_employee(request_context, params):

    employee_id = params.arguments["employee_id"]

    db = sqlite3.connect(
        "lessons/lesson01/hr.db"
    )

    row = db.execute(
        """
        SELECT employee_id, name, department
        FROM employees
        WHERE employee_id = ?
        """,
        (employee_id,)
    ).fetchone()

    db.close()

    if row is None:
        result = "Employee not found."

    else:
        result = (
            f"Employee ID: {row[0]}\n"
            f"Name: {row[1]}\n"
            f"Department: {row[2]}"
        )

    return types.CallToolResult(
        content=[
            types.TextContent(
                type="text",
                text=result
            )
        ]
    )


# -----------------------------
# Finance Database Tool
# -----------------------------

async def get_salary(request_context, params):

    employee_id = params.arguments["employee_id"]

    db = sqlite3.connect(
        "lessons/lesson01/finance.db"
    )

    row = db.execute(
        """
        SELECT employee_id, salary
        FROM salaries
        WHERE employee_id = ?
        """,
        (employee_id,)
    ).fetchone()

    db.close()

    if row is None:
        result = "Salary information not found."

    else:
        result = (
            f"Employee ID: {row[0]}\n"
            f"Salary: ₹{row[1]:,}"
        )

    return types.CallToolResult(
        content=[
            types.TextContent(
                type="text",
                text=result
            )
        ]
    )


# -----------------------------
# Find Employee by Name
# -----------------------------

async def find_employee(request_context, params):

    name = params.arguments["name"]

    db = sqlite3.connect(
        "lessons/lesson01/hr.db"
    )

    row = db.execute(
        """
        SELECT employee_id, name, department
        FROM employees
        WHERE name = ?
        """,
        (name,)
    ).fetchone()

    db.close()

    if row is None:
        result = "Employee not found."

    else:
        result = (
            f"Employee ID: {row[0]}\n"
            f"Name: {row[1]}\n"
            f"Department: {row[2]}"
        )

    return types.CallToolResult(
        content=[
            types.TextContent(
                type="text",
                text=result
            )
        ]
    )

# -----------------------------
# List MCP Tools
# -----------------------------
async def list_tools(request_context, params):

    return types.ListToolsResult(
        tools=[

            types.Tool(
                name="find_employee",
                description="Find an employee by name in the HR database.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Employee name"
                        }
                    },
                    "required": ["name"]
                }
            ),

            types.Tool(
                name="get_employee",
                description="Get employee information from the HR database.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "employee_id": {
                            "type": "integer",
                            "description": "Employee ID"
                        }
                    },
                    "required": ["employee_id"]
                }
            ),

            types.Tool(
                name="get_salary",
                description="Get employee salary from the Finance database.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "employee_id": {
                            "type": "integer",
                            "description": "Employee ID"
                        }
                    },
                    "required": ["employee_id"]
                }
            )
        ]
    )

# -----------------------------
# Route Tool Calls
# -----------------------------

async def call_tool(request_context, params):

    if params.name == "find_employee":
        return await find_employee(
            request_context,
            params
        )

    if params.name == "get_employee":
        return await get_employee(
            request_context,
            params
        )

    if params.name == "get_salary":
        return await get_salary(
            request_context,
            params
        )

    return types.CallToolResult(
        content=[
            types.TextContent(
                type="text",
                text=f"Unknown tool: {params.name}"
            )
        ],
        isError=True
    )


# -----------------------------
# Register Handlers
# -----------------------------

server = Server(
    "Employee Data Server",
    version="1.0",
    on_list_tools=list_tools,
    on_call_tool=call_tool
)


# -----------------------------
# Start MCP Server
# -----------------------------

async def main():

    async with stdio_server() as streams:

        await server.run(
            streams[0],
            streams[1],
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())