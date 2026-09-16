import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY")
)


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("human", "{question}")
])


chain = prompt | llm


question = input("Ask a question: ")

response = chain.invoke({
    "question": question
})

print("\nANSWER:")
print(response.content)