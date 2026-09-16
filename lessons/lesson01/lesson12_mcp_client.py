import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


server_params = StdioServerParameters(
    command="python",
    args=["lessons/lesson01/lesson12_mcp.py"],
)


async def main():

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            # Initialize MCP connection
            await session.initialize()

            # -----------------------------
            # Discover MCP Tools
            # -----------------------------

            tools = await session.list_tools()

            print("AVAILABLE MCP TOOLS:")

            for tool in tools.tools:
                print(f"- {tool.name}")
                print(f"  Description: {tool.description}")

            # -----------------------------
            # Call MCP Tool
            # -----------------------------

            result = await session.call_tool(
                "search_product",
                arguments={
                    "product_name": "MacBook Air"
                }
            )

            print("\nTOOL RESULT:")
            print(result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())