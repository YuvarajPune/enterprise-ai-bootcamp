import httpx

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
import asyncio


server = Server(
    "Employee API MCP Server",
    version="1.0"
)


# -----------------------------
# MCP Tool
# -----------------------------

async def get_employee(request_context, params):

    employee_id = params.arguments["employee_id"]

    # Call FastAPI
    async with httpx.AsyncClient() as client:

        response = await client.get(
            f"http://127.0.0.1:8000/employees/{employee_id}"
        )

    # API response
    data = response.json()

    return types.CallToolResult(
        content=[
            types.TextContent(
                type="text",
                text=str(data)
            )
        ]
    )


# -----------------------------
# List MCP Tools
# -----------------------------

async def find_employee(request_context, params):

    name = params.arguments["name"]

    async with httpx.AsyncClient() as client:

        response = await client.get(
            f"http://127.0.0.1:8000/employees/search/{name}"
        )

    data = response.json()

    return types.CallToolResult(
        content=[
            types.TextContent(
                type="text",
                text=str(data)
            )
        ]
    )

async def list_tools(request_context, params):

    return types.ListToolsResult(
        tools=[
            types.Tool(
    name="find_employee",
    description="Find an employee by name using the HR API.",
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
                description="Get employee information using the HR API.",
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
# Handle Tool Calls
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
# MCP Server
# -----------------------------

server = Server(
    "Employee API MCP Server",
    version="1.0",
    on_list_tools=list_tools,
    on_call_tool=call_tool
)


async def main():

    async with stdio_server() as streams:

        await server.run(
            streams[0],
            streams[1],
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())