import os

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY")
)


@tool
def calculate_total(price: float, quantity: int) -> float:
    """Calculate total price."""
    return price * quantity


@tool
def calculate_discount(amount: float, discount_percent: float) -> float:
    """Calculate the final amount after applying a discount."""
    return amount * (1 - discount_percent / 100)


agent = create_agent(
    model=llm,
    tools=[
        calculate_total,
        calculate_discount
    ],
    system_prompt="You are a helpful assistant. Use the available tools when needed."
)


question = input("Ask a question: ")

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": question
        }
    ]
})


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


print("\nFINAL ANSWER:")
print(result["messages"][-1].content)