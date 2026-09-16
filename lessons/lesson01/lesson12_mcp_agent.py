import asyncio
import os
import json

from groq import Groq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# -----------------------------
# Groq Client
# -----------------------------

llm = Groq(api_key=os.getenv("GROQ_API_KEY"))


# -----------------------------
# MCP Server Configuration
# -----------------------------

server_params = StdioServerParameters(
    command="python",
    args=["lessons/lesson01/lesson12_mcp.py"],
)


async def main():

    # -----------------------------
    # Connect MCP Client
    # -----------------------------

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()

            # -----------------------------
            # Discover MCP Tools
            # -----------------------------

            tools_result = await session.list_tools()

            print("AVAILABLE MCP TOOLS:")

            for tool in tools_result.tools:
                print(f"- {tool.name}")
                print(f"  Description: {tool.description}")

            # -----------------------------
            # Convert MCP Tools for LLM
            # -----------------------------

            tool_definitions = []

            for tool in tools_result.tools:

                tool_definitions.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema
                    }
                })

            # -----------------------------
            # User Question
            # -----------------------------

            question = input("\nWhat do you want to do? ")

            # -----------------------------
            # Ask LLM to Decide
            # -----------------------------

            response = llm.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                tools=tool_definitions
            )

            message = response.choices[0].message

            # -----------------------------
            # Check for Tool Call
            # -----------------------------

            if message.tool_calls:

                tool_call = message.tool_calls[0]

                tool_name = tool_call.function.name

                tool_arguments = json.loads(
                    tool_call.function.arguments
                )

                print("\nLLM DECIDED TO CALL:")
                print(tool_name)

                print("\nARGUMENTS:")
                print(tool_arguments)

                # -----------------------------
                # Call MCP Tool
                # -----------------------------

                result = await session.call_tool(
                    tool_name,
                    arguments=tool_arguments
                )

                tool_result = result.content[0].text

                print("\nMCP TOOL RESULT:")
                print(tool_result)

                # -----------------------------
                # Send Tool Result Back to LLM
                # -----------------------------

                final_response = llm.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {
                            "role": "user",
                            "content": question
                        },
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": message.tool_calls
                        },
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result
                        }
                    ]
                )

                print("\nFINAL ANSWER:")
                print(final_response.choices[0].message.content)

            else:

                print("\nLLM RESPONSE:")
                print(message.content)


if __name__ == "__main__":
    asyncio.run(main())