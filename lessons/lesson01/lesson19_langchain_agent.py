import os

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent


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
# Create Agent
# -----------------------------

agent = create_agent(
    model=llm,
    tools=[calculate_total],
    system_prompt="You are a helpful assistant."
)


# -----------------------------
# Run Agent
# -----------------------------

question = input("Ask a question: ")

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": question
        }
    ]
})


print("\nFINAL ANSWER:")
print(result["messages"][-1].content)