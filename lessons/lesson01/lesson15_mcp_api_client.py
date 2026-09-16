import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


server_params = StdioServerParameters(
    command="python",
    args=["lessons/lesson01/lesson15_mcp_api_server.py"],
)


async def main():

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()

            # Discover MCP tools
            tools = await session.list_tools()

            print("AVAILABLE MCP TOOLS:")

            for tool in tools.tools:
                print(f"- {tool.name}")
                print(f"  Description: {tool.description}")

            # Call MCP tool
            result = await session.call_tool(
                "get_employee",
                arguments={
                    "employee_id": 1
                }
            )

            print("\nMCP TOOL RESULT:")
            print(result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())