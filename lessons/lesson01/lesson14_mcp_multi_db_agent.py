import asyncio
import os
import json

from groq import Groq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


llm = Groq(api_key=os.getenv("GROQ_API_KEY"))


server_params = StdioServerParameters(
    command="python",
    args=["lessons/lesson01/lesson13_mcp_server.py"],
)


async def main():

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()

            # -----------------------------
            # Discover MCP Tools
            # -----------------------------

            tools_result = await session.list_tools()

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

            print("AVAILABLE MCP TOOLS:")

            for tool in tools_result.tools:
                print(f"- {tool.name}")

            # -----------------------------
            # User Question
            # -----------------------------

            question = input("\nAsk a question: ")

            messages = [
                {
                    "role": "user",
                    "content": question
                }
            ]

            # -----------------------------
            # Agent Loop
            # -----------------------------

            for step in range(5):

                print(f"\n--- AGENT STEP {step + 1} ---")

                response = llm.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=messages,
                    tools=tool_definitions
                )

                message = response.choices[0].message

                # -----------------------------
                # No More Tools
                # -----------------------------

                if not message.tool_calls:

                    print("\nFINAL ANSWER:")
                    print(message.content)
                    break

                # -----------------------------
                # Add LLM Message
                # -----------------------------

                messages.append(message)

                # -----------------------------
                # Execute Tool Calls
                # -----------------------------

                for tool_call in message.tool_calls:

                    tool_name = tool_call.function.name

                    arguments = json.loads(
                        tool_call.function.arguments
                    )

                    print(f"\nLLM → MCP TOOL:")
                    print(tool_name)
                    print(f"Arguments: {arguments}")

                    result = await session.call_tool(
                        tool_name,
                        arguments=arguments
                    )

                    tool_result = result.content[0].text

                    print("MCP RESULT:")
                    print(tool_result)

                    # -----------------------------
                    # Send Result Back to LLM
                    # -----------------------------

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })


if __name__ == "__main__":
    asyncio.run(main())