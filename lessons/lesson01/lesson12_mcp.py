import asyncio
import json

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types


server = Server("Product Server", version="1.0")


# -----------------------------
# List Available Tools
# -----------------------------

async def list_tools(request_context, params):

    return types.ListToolsResult(
        tools=[
            types.Tool(
                name="search_product",
                description="Search for a product by name.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "product_name": {
                            "type": "string",
                            "description": "Name of the product to search for."
                        }
                    },
                    "required": ["product_name"]
                }
            )
        ]
    )


# -----------------------------
# Execute Tool
# -----------------------------

async def call_tool(request_context, params):

    if params.name == "search_product":

        product_name = params.arguments["product_name"]

        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"Found product: {product_name}"
                )
            ]
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
# Register MCP Handlers
# -----------------------------

server = Server(
    "Product Server",
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