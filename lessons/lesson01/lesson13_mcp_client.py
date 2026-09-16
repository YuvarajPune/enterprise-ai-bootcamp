import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


server_params = StdioServerParameters(
    command="python",
    args=["lessons/lesson01/lesson13_mcp_server.py"],
)


async def main():

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()

            # -----------------------------
            # Discover Tools
            # -----------------------------

            tools = await session.list_tools()

            print("AVAILABLE MCP TOOLS:")

            for tool in tools.tools:
                print(f"- {tool.name}")
                print(f"  Description: {tool.description}")

            # -----------------------------
            # Get HR Information
            # -----------------------------

            employee_result = await session.call_tool(
                "get_employee",
                arguments={
                    "employee_id": 1
                }
            )

            print("\nHR DATABASE RESULT:")
            print(employee_result.content[0].text)

            # -----------------------------
            # Get Finance Information
            # -----------------------------

            salary_result = await session.call_tool(
                "get_salary",
                arguments={
                    "employee_id": 1
                }
            )

            print("\nFINANCE DATABASE RESULT:")
            print(salary_result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())