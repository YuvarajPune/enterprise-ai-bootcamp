"""
============================================================
LESSON 47 — LANGGRAPH LOOPS
============================================================

Goal:
Understand how LangGraph can repeat part of a workflow.

Key idea:

    A normal workflow:

        A → B → C → END

    A loop:

        A → B → C
            ↑   |
            └───┘

A conditional edge can send execution back to an earlier
node.

This is useful for:

    • Retry
    • Validation
    • Asking for more information
    • Iterative agent reasoning
    • Tool execution loops

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
The state keeps track of the workflow.

attempt:
    Number of times we have processed the request.

decision:
    LLM's decision about whether another attempt is needed.

result:
    Final result produced by the processing node.
"""


class AgentState(TypedDict, total=False):

    user_request: str

    attempt: int

    decision: Optional[str]

    result: Optional[str]


# ============================================================
# STEP 3 — LLM DECISION NODE
# ============================================================

async def llm_decision_node(state: AgentState):

    """
    The LLM looks at the current state and decides whether
    the workflow should continue processing or finish.

    In a real agent, this could represent reasoning such as:

        "Do I have enough information?"

        "Should I try the tool again?"

        "Do I need another iteration?"

    For this lesson, we use the attempt number so that
    the loop is predictable and easy to observe.
    """

    print("\n")
    print("=" * 60)
    print("LLM DECISION NODE")
    print("=" * 60)

    attempt = state.get(
        "attempt",
        0
    )

    prompt = f"""
You are controlling a workflow.

User request:
{state["user_request"]}

Current attempt:
{attempt}

The workflow should demonstrate a loop.

If the current attempt is less than 2,
return "continue".

If the current attempt is 2 or greater,
return "finish".

Return ONLY valid JSON:

{{
    "decision": "continue"
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

    decision = data.get(
        "decision"
    )

    print(
        "LLM DECISION:",
        decision
    )

    return {
        "decision": decision
    }


# ============================================================
# STEP 4 — PROCESS NODE
# ============================================================

def process_node(state: AgentState):

    """
    This node performs the work for the current iteration.

    Every time the loop comes back to this node,
    the attempt number is increased.

    This is what allows us to see the state changing
    during the loop.
    """

    print("\n")
    print("=" * 60)
    print("PROCESS NODE")
    print("=" * 60)

    current_attempt = state.get(
        "attempt",
        0
    )

    new_attempt = current_attempt + 1

    print(
        "Previous attempt:",
        current_attempt
    )

    print(
        "Current attempt:",
        new_attempt
    )

    return {

        "attempt":
            new_attempt,

        "result":
            f"Processing completed for attempt {new_attempt}."
    }


# ============================================================
# STEP 5 — ROUTER FUNCTION
# ============================================================

def route_after_decision(state: AgentState):

    """
    This is the routing logic.

    It reads the decision created by the LLM.

    If the LLM says:

        continue

    we send the workflow back to process_node.

    If the LLM says:

        finish

    we send the workflow to END.

    IMPORTANT:

    Just like Lesson 46:

        route_after_decision()
            =
        Routing logic

    It does not execute the next node itself.
    """

    print("\n")
    print("=" * 60)
    print("ROUTER")
    print("=" * 60)

    decision = state.get(
        "decision"
    )

    print(
        "Current decision:",
        decision
    )

    if decision == "continue":

        print(
            "ROUTE → process"
        )

        return "process"

    else:

        print(
            "ROUTE → finish"
        )

        return "finish"


# ============================================================
# STEP 6 — CREATE LANGGRAPH
# ============================================================

graph = StateGraph(
    AgentState
)


# ============================================================
# STEP 7 — REGISTER NODES
# ============================================================

graph.add_node(
    "llm_decision",
    llm_decision_node
)

graph.add_node(
    "process",
    process_node
)


# ============================================================
# STEP 8 — START → PROCESS
# ============================================================

"""
The workflow starts with process.

    START
      ↓
    process
"""

graph.add_edge(
    START,
    "process"
)


# ============================================================
# STEP 9 — PROCESS → LLM DECISION
# ============================================================

"""
After processing, we ask the LLM what should happen next.

    process
       ↓
    LLM decision
"""

graph.add_edge(
    "process",
    "llm_decision"
)


# ============================================================
# STEP 10 — CONDITIONAL EDGE / LOOP
# ============================================================

"""
This is the important part of Lesson 47.

The LLM decision determines the next path.

If:

    decision = "continue"

the workflow goes:

    llm_decision
          ↓
       process

Notice that "process" has already executed before.

We are therefore creating a LOOP.

If:

    decision = "finish"

the workflow goes:

    llm_decision
          ↓
         END
"""

graph.add_conditional_edges(

    "llm_decision",

    route_after_decision,

    {
        "process": "process",

        "finish": END
    }
)


# ============================================================
# STEP 11 — COMPILE
# ============================================================

app = graph.compile()


# ============================================================
# STEP 12 — MAIN
# ============================================================

async def main():

    print("\n")
    print("=" * 60)
    print("LESSON 47 — LANGGRAPH LOOPS")
    print("=" * 60)

    user_request = (
        "Process this request until the workflow is complete."
    )

    print("\nUSER REQUEST:")
    print(user_request)

    result = await app.ainvoke(
        {
            "user_request":
                user_request,

            "attempt":
                0
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