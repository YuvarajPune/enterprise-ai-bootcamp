"""
============================================================
LESSON 46 — LANGGRAPH CONDITIONAL EDGES
============================================================

Goal:
Understand how LangGraph chooses different paths based
on the current state.

Key concepts:

    LLM
      ↓
    Interprets the request

    route_request(state)
      ↓
    Routing logic

    graph.add_conditional_edges(...)
      ↓
    Connects routing logic to the graph

    LangGraph
      ↓
    Executes the selected node

============================================================
"""

import os
import json
import asyncio

from typing import TypedDict, Optional

from dotenv import load_dotenv
from google import genai

from langgraph.graph import StateGraph, START, END


# ============================================================
# STEP 1 — GEMINI CONFIGURATION
# ============================================================

"""
We use Gemini to classify the user's request.

Example:

    "Give me Raj's department."

LLM should return:

    employee

Another request:

    "Is MacBook Air available?"

LLM should return:

    product
"""

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found in .env"
    )

client = genai.Client(
    api_key=api_key
)


# ============================================================
# STEP 2 — LANGGRAPH STATE
# ============================================================

"""
The state is shared between LangGraph nodes.

Initially we only have:

    user_request

The LLM will add:

    request_type

The selected node will finally add:

    result
"""


class AgentState(TypedDict, total=False):

    user_request: str

    request_type: Optional[str]

    result: Optional[str]


# ============================================================
# STEP 3 — LLM INTERPRETATION NODE
# ============================================================

async def interpret_node(state: AgentState):

    """
    The LLM interprets the user's request.

    IMPORTANT:

    The LLM is deciding what the request means.

    It is NOT directly selecting or executing a
    LangGraph node.

    It puts its interpretation into the state.
    """

    print("\n")
    print("=" * 60)
    print("STEP 1 — LLM INTERPRETATION")
    print("=" * 60)

    user_request = state["user_request"]

    prompt = f"""
Classify this user request.

User request:
{user_request}

Choose exactly one:

employee
product

Return ONLY valid JSON.

Example:
{{
    "request_type": "employee"
}}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    raw_text = response.text.strip()

    print("RAW LLM RESPONSE:")
    print(raw_text)

    # Handle possible Markdown JSON fences.
    if raw_text.startswith("```"):

        lines = raw_text.splitlines()

        raw_text = "\n".join(
            line
            for line in lines
            if not line.strip().startswith("```")
        ).strip()

    data = json.loads(raw_text)

    request_type = data.get(
        "request_type"
    )

    print("\nLLM DECISION:")
    print(request_type)

    return {
        "request_type": request_type
    }


# ============================================================
# STEP 4 — ROUTER FUNCTION
# ============================================================

def route_request(state: AgentState):

    """
    ==========================================================
    ROUTING LOGIC
    ==========================================================

    This function looks at the current state and determines
    which route should be taken.

    It does NOT execute the selected node.

    For example:

        return "employee"

    only means:

        "The employee route should be taken."

    LangGraph will execute the actual node.
    """

    print("\n")
    print("=" * 60)
    print("STEP 2 — ROUTER")
    print("=" * 60)

    request_type = state.get(
        "request_type"
    )

    print(
        "Current state request_type:",
        request_type
    )

    if request_type == "employee":

        print(
            "ROUTE → employee"
        )

        return "employee"

    elif request_type == "product":

        print(
            "ROUTE → product"
        )

        return "product"

    else:

        print(
            "ROUTE → unknown"
        )

        return "unknown"


# ============================================================
# STEP 5A — EMPLOYEE NODE
# ============================================================

def employee_node(state: AgentState):

    """
    This node executes when the router returns:

        "employee"
    """

    print("\n")
    print("=" * 60)
    print("STEP 3 — EMPLOYEE NODE")
    print("=" * 60)

    result = (
        "Employee workflow selected."
    )

    print(
        "RESULT:",
        result
    )

    return {
        "result": result
    }


# ============================================================
# STEP 5B — PRODUCT NODE
# ============================================================

def product_node(state: AgentState):

    """
    This node executes when the router returns:

        "product"
    """

    print("\n")
    print("=" * 60)
    print("STEP 3 — PRODUCT NODE")
    print("=" * 60)

    result = (
        "Product workflow selected."
    )

    print(
        "RESULT:",
        result
    )

    return {
        "result": result
    }


# ============================================================
# STEP 5C — UNKNOWN NODE
# ============================================================

def unknown_node(state: AgentState):

    """
    Handles an unexpected request type.
    """

    print("\n")
    print("=" * 60)
    print("STEP 3 — UNKNOWN NODE")
    print("=" * 60)

    result = (
        "Unable to determine the request type."
    )

    print(
        "RESULT:",
        result
    )

    return {
        "result": result
    }


# ============================================================
# STEP 6 — CREATE LANGGRAPH
# ============================================================

"""
Now we create the LangGraph StateGraph.

We register the nodes first.
"""

graph = StateGraph(
    AgentState
)

graph.add_node(
    "interpret",
    interpret_node
)

graph.add_node(
    "employee",
    employee_node
)

graph.add_node(
    "product",
    product_node
)

graph.add_node(
    "unknown",
    unknown_node
)


# ============================================================
# STEP 7 — START → INTERPRET
# ============================================================

"""
This is a normal fixed edge.

START always goes to interpret.

    START
      ↓
    interpret
"""

graph.add_edge(
    START,
    "interpret"
)


# ============================================================
# STEP 8 — CONDITIONAL EDGE
# ============================================================

"""
This is the most important part of this lesson.

We tell LangGraph:

    After "interpret" finishes,
    call route_request(state).

The routing function returns:

    "employee"
    "product"
    or
    "unknown"

The dictionary maps that returned value to
an actual LangGraph node.

For example:

    route_request()
        ↓
    "employee"
        ↓
    "employee": "employee"
        ↓
    employee node executes


IMPORTANT DIFFERENCE:

    route_request(state)
        =
    Defines the routing logic.


    graph.add_conditional_edges(...)
        =
    Connects that routing logic to LangGraph.
"""

graph.add_conditional_edges(

    "interpret",

    route_request,

    {
        "employee": "employee",
        "product": "product",
        "unknown": "unknown"
    }
)


# ============================================================
# STEP 9 — CONNECT TO END
# ============================================================

"""
Once the selected node completes,
the workflow finishes.
"""

graph.add_edge(
    "employee",
    END
)

graph.add_edge(
    "product",
    END
)

graph.add_edge(
    "unknown",
    END
)


# ============================================================
# STEP 10 — COMPILE
# ============================================================

app = graph.compile()


# ============================================================
# STEP 11 — MAIN
# ============================================================

async def main():

    print("\n")
    print("=" * 60)
    print("LESSON 46 — CONDITIONAL EDGES")
    print("=" * 60)

    user_request = (
        "Give me Raj's department."
    )

    print("\nUSER REQUEST:")
    print(user_request)

    result = await app.ainvoke(
        {
            "user_request": user_request
        }
    )

    print("\n")
    print("=" * 60)
    print("FINAL LANGGRAPH STATE")
    print("=" * 60)

    print(
        json.dumps(
            result,
            indent=4
        )
    )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )