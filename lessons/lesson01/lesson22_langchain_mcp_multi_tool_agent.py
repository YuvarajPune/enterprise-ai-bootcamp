import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from langchain.agents import create_agent


# ============================================================
# SETUP / DEFINITION
# 1. LLM — reasoning engine used by the LangChain Agent
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=None  # Uses GROQ_API_KEY environment variable
)


# ============================================================
# SETUP / DEFINITION
# 2. MCP SERVER — tell the MCP Client which server to start
# ============================================================

server_params = StdioServerParameters(
    command="python",
    args=[
        "lessons/lesson01/lesson15_mcp_api_server.py"
    ],
)


async def main():

    # ========================================================
    # SETUP / DEFINITION
    # 3. MCP CLIENT — connect to the MCP Server
    # ========================================================

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()


            # =================================================
            # SETUP / DEFINITION
            # 4. DISCOVER MCP TOOLS
            # =================================================

            tools_result = await session.list_tools()

            print("MCP TOOLS:")

            for tool in tools_result.tools:
                print(f"- {tool.name}")


            # =================================================
            # SETUP / DEFINITION
            # 5. MCP TOOL WRAPPER — find_employee()
            #
            # This calls the MCP Server's find_employee tool.
            # =================================================

            async def find_employee(name: str):

                result = await session.call_tool(
                    "find_employee",
                    arguments={
                        "name": name
                    }
                )

                return result.content[0].text


            # =================================================
            # SETUP / DEFINITION
            # 6. MCP TOOL WRAPPER — get_employee()
            #
            # This calls the MCP Server's get_employee tool.
            # =================================================

            async def get_employee(employee_id: int):

                result = await session.call_tool(
                    "get_employee",
                    arguments={
                        "employee_id": employee_id
                    }
                )

                return result.content[0].text


            # =================================================
            # SETUP / DEFINITION
            # 7. CONVERT MCP TOOLS INTO LANGCHAIN TOOLS
            #
            # LangChain Agent needs LangChain-compatible tools.
            # =================================================

            find_employee_tool = StructuredTool.from_function(
                coroutine=find_employee,
                name="find_employee",
                description=(
                    "Find an employee by name using the MCP server. "
                    "Returns the employee ID and basic information."
                )
            )

            get_employee_tool = StructuredTool.from_function(
                coroutine=get_employee,
                name="get_employee",
                description=(
                    "Get complete employee information using "
                    "the employee ID through the MCP server."
                )
            )


            # =================================================
            # SETUP / DEFINITION
            # 8. LANGCHAIN AGENT
            #
            # The Agent receives both tools.
            # The LLM decides which tool to use and in what
            # sequence.
            # =================================================

            agent = create_agent(
                model=llm,
                tools=[
                    find_employee_tool,
                    get_employee_tool
                ],
                system_prompt=(
                    "You are an HR assistant. "
                    "Use the available tools to answer employee "
                    "questions accurately."
                )
            )


            # =================================================
            # RUNTIME FLOW
            # 9. USER QUESTION
            #
            # Example:
            # "What department does Raj work in?"
            # =================================================

            question = input("\nAsk a question: ")


            # =================================================
            # RUNTIME FLOW
            # 10. AGENT EXECUTION
            #
            # Expected flow:
            #
            # User
            #   ↓
            # LangChain Agent
            #   ↓
            # LLM
            #   ↓
            # find_employee("Raj")
            #   ↓
            # employee_id = 1
            #   ↓
            # LLM
            #   ↓
            # get_employee(1)
            #   ↓
            # Employee information
            #   ↓
            # LLM
            #   ↓
            # Final Answer
            # =================================================

            result = await agent.ainvoke({
                "messages": [
                    {
                        "role": "user",
                        "content": question
                    }
                ]
            })


            # =================================================
            # RUNTIME FLOW
            # 11. SHOW AGENT TOOL EXECUTION
            #
            # This lets us see which MCP-backed tools were used.
            # =================================================

            print("\n--- AGENT EXECUTION ---")

            for message in result["messages"]:

                if hasattr(message, "tool_calls") and message.tool_calls:

                    for tool_call in message.tool_calls:

                        print("\nLLM → TOOL:")
                        print(tool_call["name"])

                        print("ARGUMENTS:")
                        print(tool_call["args"])

                elif message.type == "tool":

                    print("TOOL RESULT:")
                    print(message.content)


            # =================================================
            # RUNTIME FLOW
            # 12. FINAL ANSWER
            # =================================================

            print("\nFINAL ANSWER:")
            print(result["messages"][-1].content)


# ============================================================
# 13. PROGRAM START
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())