import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from langchain.agents import create_agent


# ============================================================
# 1. LLM — the Agent's reasoning engine
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=None  # Uses GROQ_API_KEY environment variable
)


# ============================================================
# 2. MCP SERVER CONNECTION DETAILS
#    This tells the MCP Client which MCP Server to connect to.
# ============================================================

server_params = StdioServerParameters(
    command="python",
    args=[
        "lessons/lesson01/lesson15_mcp_api_server.py"
    ],
)


async def main():

    # ========================================================
    # 3. MCP CLIENT — connects to the MCP Server
    # ========================================================

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()


            # =================================================
            # 4. MCP TOOL DISCOVERY
            #    Discover tools exposed by the MCP Server
            # =================================================

            tools_result = await session.list_tools()

            print("MCP TOOLS:")

            for tool in tools_result.tools:
                print(f"- {tool.name}")


            # =================================================
            # 5. MCP TOOL WRAPPER
            #    Wrap the MCP tool so LangChain can use it
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
            # 6. LANGCHAIN TOOL
            #    Expose the MCP capability as a LangChain Tool
            # =================================================

            find_employee_tool = StructuredTool.from_function(
                coroutine=find_employee,
                name="find_employee",
                description="Find an employee by name using the MCP server."
            )


            # =================================================
            # 7. LANGCHAIN AGENT
            #    The Agent decides when to use the tool
            # =================================================

            agent = create_agent(
                model=llm,
                tools=[find_employee_tool],
                system_prompt=(
                    "You are an HR assistant. "
                    "Use the available tools when needed "
                    "to answer employee questions."
                )
            )


            # =================================================
            # 8. USER QUESTION
            #    This starts the Agent flow
            # =================================================

            question = input("\nAsk a question: ")


            # =================================================
            # 9. AGENT EXECUTION
            #
            #    Agent
            #      ↓
            #    LLM
            #      ↓
            #    LangChain Tool
            #      ↓
            #    MCP Client
            #      ↓
            #    MCP Server
            #      ↓
            #    FastAPI
            #      ↓
            #    HR Database
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
            # 10. FINAL ANSWER
            #     LLM produces the answer using the tool result
            # =================================================

            print("\nFINAL ANSWER:")
            print(result["messages"][-1].content)


# ============================================================
# 11. START PROGRAM
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())