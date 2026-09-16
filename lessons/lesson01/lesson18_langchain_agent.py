import os

from langchain_groq import ChatGroq
from langchain_core.tools import tool


# -----------------------------
# LLM
# -----------------------------

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY")
)


# -----------------------------
# Tool
# -----------------------------

@tool
def calculate_total(price: float, quantity: int) -> float:
    """Calculate total price from price and quantity."""
    return price * quantity


# -----------------------------
# Give tool to LLM
# -----------------------------

llm_with_tools = llm.bind_tools(
    [calculate_total]
)


# -----------------------------
# User question
# -----------------------------

question = input("Ask a question: ")

messages = [
    ("human", question)
]


# -----------------------------
# Agent loop
# -----------------------------

while True:

    response = llm_with_tools.invoke(messages)

    messages.append(response)

    if not response.tool_calls:

        print("\nFINAL ANSWER:")
        print(response.content)
        break

    for tool_call in response.tool_calls:

        print("\nLLM DECIDED TO CALL:")
        print(tool_call["name"])

        print("ARGUMENTS:")
        print(tool_call["args"])

        if tool_call["name"] == "calculate_total":

            result = calculate_total.invoke(
                tool_call["args"]
            )

            print("TOOL RESULT:")
            print(result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": str(result)
                }
            )