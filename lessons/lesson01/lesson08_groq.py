# -----------------------------
# Import Libraries
# -----------------------------

import os
import json
from groq import Groq


# -----------------------------
# Define Tool Function
# -----------------------------

def search_product(product_name):
    return f"Found product: {product_name}"

def get_weather(question):
    return "Tomorrow will be 28°C with light rain."

# -----------------------------
# Define Tool for Groq
# -----------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_product",
            "description": "Search for a product.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "Name of the product to search for."
                    }
                },
                "required": ["product_name"]
            }
        }
    }
]
# -----------------------------
# Create Groq Client
# -----------------------------

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)


# -----------------------------
# Ask User a Question
# -----------------------------

question = input("Ask me something: ")


# -----------------------------
# Send Question to Groq
# -----------------------------

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": question
        }
    ]
)

# -----------------------------
# Ask Groq to Create a Plan
# -----------------------------

planning_prompt = f"""
Create a plan for this user request:

{question}

Available tools:

1. search_product
   Arguments: product_name
   Description: Search for a product.

2. get_weather
   Arguments: question
   Description: Get weather information.

Return ONLY valid JSON.

Each step must contain:
- tool
- args

Example:

[
    {{
        "tool": "search_product",
        "args": {{
            "product_name": "MacBook Air"
        }}
    }},
    {{
        "tool": "get_weather",
        "args": {{
            "question": "weather tomorrow"
        }}
    }}
]
"""

# -----------------------------
# Ask Groq to Create a Plan
# -----------------------------

planning_prompt = f"""
Create a plan for this user request:

{question}

Available tools:

1. search_product
   Arguments: product_name
   Description: Search for a product.

2. get_weather
   Arguments: question
   Description: Get weather information.

Return ONLY valid JSON.

Each step must contain:
- tool
- args

Example:

[
    {{
        "tool": "search_product",
        "args": {{
            "product_name": "MacBook Air"
        }}
    }},
    {{
        "tool": "get_weather",
        "args": {{
            "question": "weather tomorrow"
        }}
    }}
]
"""

plan_response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": planning_prompt
        }
    ]
)


# -----------------------------
# Convert Plan JSON to Python
# -----------------------------

plan_text = plan_response.choices[0].message.content

# Remove Markdown code fences
plan_text = plan_text.replace("```json", "").replace("```", "").strip()

# Convert JSON text to Python
plan = json.loads(plan_text)

print("PLAN:")
print(plan)

# -----------------------------
# Convert Plan JSON to Python
# -----------------------------

plan_text = plan_response.choices[0].message.content

# Remove Markdown code fences
plan_text = plan_text.replace("```json", "").replace("```", "").strip()

# Convert JSON text to Python
plan = json.loads(plan_text)

print("PLAN:")
print(plan)

# -----------------------------
# Tool Registry
# -----------------------------

tools = {
    "search_product": search_product,
    "get_weather": get_weather
}




# -----------------------------
# Execute the Plan
# -----------------------------

for step in plan:

    # Get the tool name
    tool_name = step["tool"]

    # Get the tool arguments
    tool_arguments = step["args"]

    print("Executing:", tool_name)
    print("Arguments:", tool_arguments)

    # Find the function from the Tool Registry
    function = tools[tool_name]

    # Execute the function
    answer = function(**tool_arguments)

    print("Result:", answer)


# -----------------------------
# Find Function from Tool Registry
# -----------------------------


