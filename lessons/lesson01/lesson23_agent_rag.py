# ============================================================
# LESSON 23 — AGENT + RAG
#
# Goal:
# Connect a LangChain Agent to a RAG Tool.
#
# RUNTIME ARCHITECTURE:
#
# User
#   ↓
# LangChain Agent
#   ↓
# LLM decides to use search_knowledge()
#   ↓
# RAG Tool
#   ↓
# Retriever
#   ↓
# Chroma Vector DB
#   ↓
# Relevant documents
#   ↓
# LLM
#   ↓
# Final Answer
# ============================================================


# ============================================================
# SETUP / DEFINITION
# 1. IMPORTS
# ============================================================

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
from langchain.agents import create_agent


# ============================================================
# SETUP / DEFINITION
# 2. LLM — AGENT'S REASONING ENGINE
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=None
)


# ============================================================
# SETUP / DEFINITION
# 3. KNOWLEDGE BASE
#
# These represent enterprise documents.
# ============================================================

documents = [
    Document(
        page_content="Employees receive 24 days of annual leave per year.",
        metadata={"topic": "leave policy"}
    ),
    Document(
        page_content="Employees can work from home up to two days per week.",
        metadata={"topic": "work from home"}
    ),
    Document(
        page_content="Business travel expenses must be submitted within 30 days.",
        metadata={"topic": "travel policy"}
    ),
]


# ============================================================
# SETUP / DEFINITION
# 4. EMBEDDING MODEL
#
# Converts documents and questions into vectors.
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# SETUP / DEFINITION
# 5. VECTOR DATABASE — CHROMA
#
# Documents
#    ↓
# Embeddings
#    ↓
# Chroma
# ============================================================

vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="company_knowledge"
)


# ============================================================
# SETUP / DEFINITION
# 6. RETRIEVER
#
# Question
#    ↓
# Semantic similarity search
#    ↓
# Ranked documents
#    ↓
# Top 2 results
# ============================================================

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)


# ============================================================
# SETUP / DEFINITION
# 7. RAG TOOL
#
# This is the capability exposed to the Agent.
#
# Agent does NOT directly access Chroma.
#
# Agent
#   ↓
# search_knowledge()
#   ↓
# Retriever
#   ↓
# Vector DB
# ============================================================

@tool
def search_knowledge(question: str) -> str:
    """Search the company knowledge base for relevant information."""

    results = retriever.invoke(question)

    if not results:
        return "No relevant information found."

    return "\n".join(
        document.page_content
        for document in results
    )


# ============================================================
# SETUP / DEFINITION
# 8. LANGCHAIN AGENT
#
# The Agent receives the RAG Tool.
#
# The LLM decides:
# "Do I need to search the knowledge base?"
# ============================================================

agent = create_agent(
    model=llm,
    tools=[search_knowledge],
    system_prompt=(
        "You are a helpful company policy assistant. "
        "Use the search_knowledge tool when the question "
        "requires information from company documents."
    )
)


# ============================================================
# RUNTIME FLOW
# 9. USER QUESTION
# ============================================================

question = input("\nAsk a question: ")


# ============================================================
# RUNTIME FLOW
# 10. AGENT EXECUTION
#
# User
#   ↓
# Agent
#   ↓
# LLM
#   ↓
# search_knowledge()
#   ↓
# Retriever
#   ↓
# Chroma
#   ↓
# Retrieved context
#   ↓
# LLM
#   ↓
# Answer
# ============================================================

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": question
        }
    ]
})


# ============================================================
# RUNTIME FLOW
# 11. SHOW TOOL EXECUTION
#
# This lets us see whether the Agent actually used RAG.
# ============================================================

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


# ============================================================
# RUNTIME FLOW
# 12. FINAL ANSWER
# ============================================================

print("\nFINAL ANSWER:")
print(result["messages"][-1].content)