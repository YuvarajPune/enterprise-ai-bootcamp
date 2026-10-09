"""
============================================================
LESSON 45.9 — PARALLEL MCP + FAILURE RECOVERY
============================================================

PURPOSE
------------------------------------------------------------
This lesson combines several LangGraph + MCP concepts that
we learned in Lessons 45.1 through 45.8.

We will demonstrate:

    1. LLM interpretation
    2. LangGraph state
    3. Fan-out
    4. Multiple MCP servers
    5. Parallel MCP execution
    6. Independent branch failures
    7. Fan-in
    8. State-aware failure analysis
    9. LLM-assisted recovery decision
    10. Recovery of only the failed branch
    11. Final LLM response

IMPORTANT ARCHITECTURE
------------------------------------------------------------

                    USER REQUEST
                         |
                         v
                  LLM INTERPRETATION
                         |
                    LANGGRAPH
                         |
                  +------+------+
                  |             |
                  v             v
           EMPLOYEE MCP    PRODUCT MCP
                  |             |
               SUCCESS        FAILURE
                  |             |
                  +------+------+
                         |
                       FAN-IN
                         |
                         v
                 RECOVERY ANALYSIS
                         |
                         v
                  RECOVERY ACTION
                         |
                         v
                    FINAL LLM

KEY LEARNING
------------------------------------------------------------

A failure in one parallel branch should NOT automatically
destroy successful information from another branch.

Example:

    Employee MCP  -> SUCCESS
    Product MCP   -> FAILURE

The employee information should remain available in the
LangGraph state.

We then recover only the failed product branch.

This is an important enterprise agent design pattern.

============================================================
"""


# ============================================================
# IMPORTS
# ============================================================

import asyncio
import json
import os

from typing import TypedDict, Optional

from dotenv import load_dotenv
from google import genai

from langgraph.graph import StateGraph, START, END

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ============================================================
# ENVIRONMENT / LLM CONFIGURATION
# ============================================================

"""
We use Gemini as the LLM for this lesson.

The API key is loaded from the .env file.

Example .env:

    GEMINI_API_KEY=your_key_here

The MCP servers themselves do NOT need the Gemini key.

Only this LangGraph application uses the LLM.
"""

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:

    raise ValueError(
        "GEMINI_API_KEY not found. "
        "Please configure GEMINI_API_KEY in your .env file."
    )


client = genai.Client(
    api_key=api_key
)


# ============================================================
# LANGGRAPH STATE
# ============================================================

"""
The LangGraph State is the shared memory of the workflow.

Every node receives the current state.

A node can return additional / updated fields.

The important point in this lesson is that the state keeps
the results of BOTH parallel branches.

For example:

    employee       = successful result
    product        = None
    product_error  = failure information

The successful employee information is therefore preserved
even though the product branch failed.
"""


class AgentState(TypedDict, total=False):

    # --------------------------------------------------------
    # Original user request
    # --------------------------------------------------------

    user_request: str


    # --------------------------------------------------------
    # LLM interpretation
    # --------------------------------------------------------

    employee_name: Optional[str]

    employee_id: Optional[int]

    product_name: Optional[str]


    # --------------------------------------------------------
    # Employee MCP result
    # --------------------------------------------------------

    employee: Optional[str]

    employee_status: Optional[str]

    employee_error: Optional[str]


    # --------------------------------------------------------
    # Product MCP result
    # --------------------------------------------------------

    product: Optional[str]

    product_status: Optional[str]

    product_error: Optional[str]


    # --------------------------------------------------------
    # Recovery information
    # --------------------------------------------------------

    recovery_needed: Optional[bool]

    recovery_recommendation: Optional[str]

    recovery_reason: Optional[str]


    # --------------------------------------------------------
    # Recovery execution
    # --------------------------------------------------------

    recovery_result: Optional[str]


    # --------------------------------------------------------
    # Final answer
    # --------------------------------------------------------

    final_answer: Optional[str]


# ============================================================
# HELPER FUNCTION
# ============================================================

def clean_json_response(response: str) -> str:

    """
    Gemini may sometimes return JSON inside Markdown fences.

    Example:

        ```json
        {
            "employee_name": "Raj"
        }
        ```

    json.loads() cannot directly parse that.

    This helper removes the Markdown fences before we
    deserialize the JSON.
    """

    response = response.strip()

    if response.startswith("```"):

        lines = response.splitlines()

        cleaned_lines = []

        for line in lines:

            if line.strip().startswith("```"):
                continue

            cleaned_lines.append(line)

        response = "\n".join(
            cleaned_lines
        ).strip()

    return response


# ============================================================
# STEP 1 — LLM INTERPRETATION
# ============================================================

async def interpret_node(state: AgentState):

    """
    The first LangGraph node.

    Its responsibility is NOT to call MCP.

    It simply converts the natural-language user request
    into structured information that later nodes can use.

    Example:

        User:
        "Give me Raj's department and tell me whether
         MacBook Air is available."

    becomes:

        {
            "employee_name": "Raj",
            "product_name": "MacBook Air"
        }

    This information is stored in LangGraph state.
    """

    print("\n")
    print("=" * 60)
    print("STEP 1 — LLM INTERPRETATION")
    print("=" * 60)

    prompt = f"""
Interpret the following business request.

User request:
{state["user_request"]}

Extract:

1. employee_name
2. product_name

Return ONLY valid JSON.

Example:

{{
    "employee_name": "Raj",
    "product_name": "MacBook Air"
}}

If a value is not present, use null.

Do not return Markdown.
Do not return explanation.
"""

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt
    )

    raw_response = interaction.output_text

    response = clean_json_response(
        raw_response
    )

    print(response)

    data = json.loads(response)

    return {

        "employee_name":
            data.get("employee_name"),

        "product_name":
            data.get("product_name")
    }


# ============================================================
# STEP 2A — EMPLOYEE MCP BRANCH
# ============================================================

async def employee_node(state: AgentState):

    """
    This is one of the two parallel branches.

    It connects to the existing Employee MCP server created
    earlier in Lesson 13.

    Existing server:

        lessons/lesson01/lesson13_mcp_server.py

    Available tools:

        find_employee
        get_employee
        get_salary

    For this lesson we use:

        find_employee
        get_employee

    IMPORTANT:

    This node catches its own exceptions.

    Therefore, if the Employee MCP fails, it returns a
    FAILED status into LangGraph state instead of crashing
    the entire graph.
    """

    print("\n")
    print("[PARALLEL BRANCH] EMPLOYEE MCP STARTED")

    try:

        # ----------------------------------------------------
        # Start existing Employee MCP server
        # ----------------------------------------------------

        server_params = StdioServerParameters(

            command="python",

            args=[
                "lessons/lesson01/lesson13_mcp_server.py"
            ]
        )


        # ----------------------------------------------------
        # Create MCP stdio connection
        # ----------------------------------------------------

        async with stdio_client(
            server_params
        ) as (read, write):


            # ------------------------------------------------
            # Create MCP client session
            # ------------------------------------------------

            async with ClientSession(
                read,
                write
            ) as session:


                # --------------------------------------------
                # Initialize MCP connection
                # --------------------------------------------

                await session.initialize()


                # --------------------------------------------
                # Discover available tools
                # --------------------------------------------

                tools_result = await session.list_tools()

                employee_tools = [
                    tool.name
                    for tool in tools_result.tools
                ]

                print(
                    "Employee MCP tools:",
                    employee_tools
                )


                # --------------------------------------------
                # Find employee by name
                # --------------------------------------------

                print(
                    "\n[MCP CLIENT] "
                    "Calling find_employee"
                )

                result = await session.call_tool(

                    "find_employee",

                    arguments={
                        "name":
                            state["employee_name"]
                    }
                )


                employee_lookup = (
                    result.content[0].text
                )

                print(
                    "\nfind_employee RESULT:"
                )

                print(employee_lookup)


                # --------------------------------------------
                # Extract employee ID
                # --------------------------------------------

                employee_id = None

                for line in employee_lookup.splitlines():

                    if line.startswith(
                        "Employee ID:"
                    ):

                        employee_id = int(
                            line.split(":")[1].strip()
                        )


                # --------------------------------------------
                # Get complete employee information
                # --------------------------------------------

                employee = None

                if employee_id is not None:

                    print(
                        "\n[MCP CLIENT] "
                        "Calling get_employee"
                    )

                    result = await session.call_tool(

                        "get_employee",

                        arguments={
                            "employee_id":
                                employee_id
                        }
                    )

                    employee = (
                        result.content[0].text
                    )

                    print(
                        "\nGET EMPLOYEE RESULT:"
                    )

                    print(employee)


        # ----------------------------------------------------
        # Employee MCP completed successfully
        # ----------------------------------------------------

        print(
            "\n[PARALLEL BRANCH] "
            "EMPLOYEE MCP COMPLETED"
        )

        return {

            "employee": employee,

            "employee_id":
                employee_id,

            "employee_status":
                "SUCCESS",

            "employee_error":
                None
        }


    except Exception as e:

        # ----------------------------------------------------
        # Important:
        #
        # We capture the failure inside the state instead
        # of allowing the entire LangGraph execution to fail.
        # ----------------------------------------------------

        error_message = str(e)

        print(
            "\n[PARALLEL BRANCH] "
            "EMPLOYEE MCP FAILED"
        )

        print(
            "ERROR:",
            error_message
        )

        return {

            "employee": None,

            "employee_id": None,

            "employee_status":
                "FAILED",

            "employee_error":
                error_message
        }


# ============================================================
# STEP 2B — PRODUCT MCP BRANCH
# ============================================================

async def product_node(state: AgentState):

    """
    This is the second parallel branch.

    It connects to the existing Product MCP server from
    Lesson 12.

    Existing server:

        lessons/lesson01/lesson12_mcp.py

    Available tool:

        search_product

    We intentionally call a NON-EXISTENT tool:

        search_product_unavailable

    This creates a controlled MCP failure.

    The purpose is to demonstrate that the failure of this
    branch does not destroy the successful Employee branch.
    """

    print("\n")
    print("[PARALLEL BRANCH] PRODUCT MCP STARTED")

    try:

        # ----------------------------------------------------
        # Start Product MCP server
        # ----------------------------------------------------

        server_params = StdioServerParameters(

            command="python",

            args=[
                "lessons/lesson01/lesson12_mcp.py"
            ]
        )


        # ----------------------------------------------------
        # Connect to Product MCP
        # ----------------------------------------------------

        async with stdio_client(
            server_params
        ) as (read, write):


            async with ClientSession(
                read,
                write
            ) as session:


                # --------------------------------------------
                # Initialize MCP
                # --------------------------------------------

                await session.initialize()


                # --------------------------------------------
                # Discover tools
                # --------------------------------------------

                tools_result = await session.list_tools()

                product_tools = [
                    tool.name
                    for tool in tools_result.tools
                ]

                print(
                    "Product MCP tools:",
                    product_tools
                )


                # --------------------------------------------
                # Intentionally call unavailable tool
                # --------------------------------------------

                print(
                    "\n[MCP CLIENT] "
                    "Calling intentionally unavailable tool"
                )

                result = await session.call_tool(

                    "search_product_unavailable",

                    arguments={
                        "product_name":
                            state["product_name"]
                    }
                )


                product_result = (
                    result.content[0].text
                )

                print(
                    "\nPRODUCT MCP RESULT:"
                )

                print(product_result)


                # --------------------------------------------
                # Convert the MCP failure into a Python
                # exception so that the branch can record
                # failure state.
                # --------------------------------------------

                if getattr(
                    result,
                    "isError",
                    False
                ):

                    raise RuntimeError(
                        product_result
                    )


                if product_result.startswith(
                    "Unknown tool:"
                ):

                    raise RuntimeError(
                        product_result
                    )


                # ------------------------------------------------
                # Successful product result
                # ------------------------------------------------

                print(
                    "\n[PARALLEL BRANCH] "
                    "PRODUCT MCP COMPLETED"
                )

                return {

                    "product":
                        product_result,

                    "product_status":
                        "SUCCESS",

                    "product_error":
                        None
                }


    except Exception as e:

        # ----------------------------------------------------
        # Capture only the Product branch failure.
        # ----------------------------------------------------

        error_message = str(e)

        print(
            "\n[PARALLEL BRANCH] "
            "PRODUCT MCP FAILED"
        )

        print(
            "ERROR:",
            error_message
        )

        return {

            "product": None,

            "product_status":
                "FAILED",

            "product_error":
                error_message
        }


# ============================================================
# STEP 3 — FAN-IN
# ============================================================

async def fan_in_node(state: AgentState):

    """
    FAN-IN combines the results of the two parallel branches.

    Before this node:

        Employee branch
                 |
                 v
            employee

        Product branch
                 |
                 v
             product

    At FAN-IN we can see both results.

    Example:

        employee_status = SUCCESS
        product_status  = FAILED

    The important concept:

        We DO NOT discard employee.

    The successful result remains in the shared state.
    """

    print("\n")
    print("=" * 60)
    print("STEP 3 — FAN-IN")
    print("=" * 60)

    print(
        "\nEMPLOYEE BRANCH STATUS:"
    )

    print(
        state.get("employee_status")
    )


    print(
        "\nPRODUCT BRANCH STATUS:"
    )

    print(
        state.get("product_status")
    )


    # --------------------------------------------------------
    # Show the successful state
    # --------------------------------------------------------

    if state.get(
        "employee_status"
    ) == "SUCCESS":

        print(
            "\nEMPLOYEE INFORMATION PRESERVED:"
        )

        print(
            state.get("employee")
        )


    # --------------------------------------------------------
    # Detect product failure
    # --------------------------------------------------------

    if state.get(
        "product_status"
    ) == "FAILED":

        recovery_needed = True

        print(
            "\nPRODUCT BRANCH FAILED."
        )

        print(
            "Recovery is required."
        )

    else:

        recovery_needed = False

        print(
            "\nPRODUCT BRANCH SUCCEEDED."
        )

        print(
            "No recovery required."
        )


    return {

        "recovery_needed":
            recovery_needed
    }


# ============================================================
# STEP 4 — LLM RECOVERY ANALYSIS
# ============================================================

async def recovery_analysis_node(
    state: AgentState
):

    """
    The LLM now receives the combined state produced by
    the parallel branches.

    This is important.

    The LLM is no longer looking only at the original request.

    It can see:

        - employee result
        - employee status
        - product result
        - product status
        - product error

    It then recommends what should happen next.

    Allowed decisions:

        CONTINUE_PARTIALLY
        FALLBACK
        STOP

    The LLM is recommending.

    It is NOT directly executing recovery.
    """

    print("\n")
    print("=" * 60)
    print("STEP 4 — LLM RECOVERY ANALYSIS")
    print("=" * 60)


    # --------------------------------------------------------
    # No recovery needed
    # --------------------------------------------------------

    if not state.get(
        "recovery_needed"
    ):

        print(
            "No recovery analysis required."
        )

        return {

            "recovery_recommendation":
                "CONTINUE",

            "recovery_reason":
                "Both parallel branches succeeded."
        }


    # --------------------------------------------------------
    # Ask LLM to analyze the partial state
    # --------------------------------------------------------

    prompt = f"""
You are an enterprise workflow recovery planner.

Analyze the current LangGraph state.

USER REQUEST:
{state["user_request"]}

EMPLOYEE STATUS:
{state.get("employee_status")}

EMPLOYEE INFORMATION:
{state.get("employee")}

PRODUCT STATUS:
{state.get("product_status")}

PRODUCT REQUESTED:
{state.get("product_name")}

PRODUCT ERROR:
{state.get("product_error")}

Choose ONE recovery recommendation:

1. CONTINUE_PARTIALLY
   Use when the successful information is sufficient to
   answer at least part of the request.

2. FALLBACK
   Use when the failed branch should be recovered using
   another approved capability.

3. STOP
   Use when the workflow cannot safely continue.

Return ONLY valid JSON:

{{
    "recommendation": "CONTINUE_PARTIALLY",
    "reason": "..."
}}

Do not return Markdown.
Do not invent product information.
"""


    interaction = client.interactions.create(

        model="gemini-3.6-flash",

        input=prompt
    )


    response = clean_json_response(
        interaction.output_text
    )


    print(
        "\nRAW LLM RECOVERY RESPONSE:"
    )

    print(response)


    data = json.loads(
        response
    )


    recommendation = data.get(
        "recommendation"
    )

    reason = data.get(
        "reason"
    )


    print(
        "\nLLM RECOVERY RECOMMENDATION:"
    )

    print(
        recommendation
    )


    print(
        "\nLLM RECOVERY REASON:"
    )

    print(
        reason
    )


    return {

        "recovery_recommendation":
            recommendation,

        "recovery_reason":
            reason
    }


# ============================================================
# STEP 5 — RECOVERY EXECUTION
# ============================================================

async def recovery_node(
    state: AgentState
):

    """
    This node executes the recovery strategy recommended
    by the LLM.

    IMPORTANT:

    The recovery is applied ONLY to the failed branch.

    Employee information is already successful.

    Therefore:

        Employee → keep existing result

        Product  → attempt recovery

    This demonstrates STATE-AWARE RECOVERY.
    """

    print("\n")
    print("=" * 60)
    print("STEP 5 — RECOVERY EXECUTION")
    print("=" * 60)


    recommendation = state.get(
        "recovery_recommendation"
    )


    # --------------------------------------------------------
    # Both branches succeeded
    # --------------------------------------------------------

    if recommendation == "CONTINUE":

        print(
            "Both parallel branches succeeded."
        )

        print(
            "No recovery required."
        )

        return {}


    # --------------------------------------------------------
    # Partial continuation
    # --------------------------------------------------------

    if recommendation == "CONTINUE_PARTIALLY":

        print(
            "LLM recommended partial continuation."
        )

        print(
            "Preserving successfully retrieved employee "
            "information."
        )

        return {

            "recovery_result":
                "Continued using successfully "
                "retrieved information."
        }


    # --------------------------------------------------------
    # Stop
    # --------------------------------------------------------

    if recommendation == "STOP":

        print(
            "LLM recommended STOP."
        )

        return {

            "recovery_result":
                "Workflow stopped because recovery "
                "was not considered safe."
        }


    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if recommendation == "FALLBACK":

        print(
            "LLM recommended FALLBACK."
        )

        print(
            "Recovering ONLY the failed product branch."
        )


        # ----------------------------------------------------
        # IMPORTANT:
        #
        # This is a simulated fallback capability.
        #
        # We are not claiming that the product is available.
        #
        # We are only demonstrating that a failed branch
        # can be replaced by another recovery mechanism.
        # ----------------------------------------------------

        fallback_result = (

            f"Fallback product lookup completed "
            f"for {state['product_name']}."
        )


        print(
            "\nFALLBACK RESULT:"
        )

        print(
            fallback_result
        )


        return {

            "recovery_result":
                fallback_result
        }


    # --------------------------------------------------------
    # Unknown recommendation
    # --------------------------------------------------------

    print(
        "Unknown recovery recommendation."
    )

    return {

        "recovery_result":
            "Recovery could not be executed."
    }


# ============================================================
# STEP 6 — FINAL LLM
# ============================================================

async def final_node(
    state: AgentState
):

    """
    The final LLM converts the complete LangGraph state
    into a business-friendly answer.

    The LLM must distinguish between:

        1. Successfully retrieved information
        2. Failed information
        3. Recovery information

    Most importantly:

        "Fallback lookup completed"

    does NOT automatically mean:

        "Product is available"

    We therefore instruct the LLM not to invent availability.
    """

    print("\n")
    print("=" * 60)
    print("STEP 6 — FINAL LLM")
    print("=" * 60)


    prompt = f"""
Generate the final business answer using ONLY the
information contained in the workflow state.

USER REQUEST:
{state["user_request"]}

EMPLOYEE:
{state.get("employee")}

EMPLOYEE STATUS:
{state.get("employee_status")}

PRODUCT REQUESTED:
{state.get("product_name")}

PRODUCT RESULT:
{state.get("product")}

PRODUCT STATUS:
{state.get("product_status")}

PRODUCT ERROR:
{state.get("product_error")}

RECOVERY RECOMMENDATION:
{state.get("recovery_recommendation")}

RECOVERY REASON:
{state.get("recovery_reason")}

RECOVERY RESULT:
{state.get("recovery_result")}

IMPORTANT RULES:

- Preserve successful employee information.
- Do not invent product information.
- Do not say a product is available unless the state
  explicitly confirms availability.
- Clearly distinguish original MCP failure from recovery.
- Give a concise business answer.
"""


    interaction = client.interactions.create(

        model="gemini-3.6-flash",

        input=prompt
    )


    answer = interaction.output_text


    print(
        "\nFINAL BUSINESS ANSWER:"
    )

    print(
        answer
    )


    return {

        "final_answer":
            answer
    }


# ============================================================
# BUILD LANGGRAPH
# ============================================================

"""
Now we construct the LangGraph workflow.

The important part is the FAN-OUT.

After the interpretation node:

        interpret
             |
       +-----+-----+
       |           |
       v           v
   employee     product
       |           |
       +-----+-----+
             |
           fan_in

LangGraph runs the independent branches as parallel
graph branches.
"""


graph = StateGraph(
    AgentState
)


# ============================================================
# REGISTER NODES
# ============================================================

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
    "fan_in",
    fan_in_node
)


graph.add_node(
    "recovery_analysis",
    recovery_analysis_node
)


graph.add_node(
    "recovery",
    recovery_node
)


graph.add_node(
    "final",
    final_node
)


# ============================================================
# START → INTERPRET
# ============================================================

graph.add_edge(
    START,
    "interpret"
)


# ============================================================
# FAN-OUT
# ============================================================

"""
This is the FAN-OUT.

The interpretation result is sent to TWO independent branches.

Branch 1:

    interpret → employee

Branch 2:

    interpret → product

These branches do not depend on each other.
"""


graph.add_edge(
    "interpret",
    "employee"
)


graph.add_edge(
    "interpret",
    "product"
)


# ============================================================
# EACH BRANCH → FAN-IN
# ============================================================

"""
Both branches eventually arrive at FAN-IN.

Employee:

    employee → fan_in

Product:

    product → fan_in

LangGraph waits for the parallel branches before executing
the fan_in node.
"""


graph.add_edge(
    "employee",
    "fan_in"
)


graph.add_edge(
    "product",
    "fan_in"
)


# ============================================================
# FAN-IN → RECOVERY ANALYSIS
# ============================================================

graph.add_edge(
    "fan_in",
    "recovery_analysis"
)


# ============================================================
# RECOVERY ANALYSIS → RECOVERY
# ============================================================

graph.add_edge(
    "recovery_analysis",
    "recovery"
)


# ============================================================
# RECOVERY → FINAL
# ============================================================

graph.add_edge(
    "recovery",
    "final"
)


# ============================================================
# FINAL → END
# ============================================================

graph.add_edge(
    "final",
    END
)


# ============================================================
# COMPILE GRAPH
# ============================================================

app = graph.compile()


# ============================================================
# MAIN
# ============================================================

async def main():

    print("\n")
    print("=" * 60)
    print("LESSON 45.9 — PARALLEL MCP + FAILURE RECOVERY")
    print("=" * 60)


    # --------------------------------------------------------
    # User request
    # --------------------------------------------------------

    user_request = (
        "Give me Raj's department and tell me "
        "whether MacBook Air is available."
    )


    # --------------------------------------------------------
    # Start LangGraph
    #
    # Only the initial user_request is provided.
    #
    # All other state values are created by nodes.
    # --------------------------------------------------------

    result = await app.ainvoke(

        {
            "user_request":
                user_request
        }
    )


    # ========================================================
    # FINAL LANGGRAPH STATE
    # ========================================================

    print("\n")
    print("=" * 60)
    print("FINAL LANGGRAPH STATE")
    print("=" * 60)


    print(
        json.dumps(
            result,
            indent=4,
            ensure_ascii=False
        )
    )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )


"""
============================================================
LESSON 45.9 LEARNING SUMMARY
============================================================

The important architecture is:

                LLM INTERPRETATION
                       |
                +------+------+
                |             |
                v             v
          EMPLOYEE MCP    PRODUCT MCP
             SUCCESS        FAILURE
                |             |
                +------+------+
                       |
                     FAN-IN
                       |
              LLM RECOVERY ANALYSIS
                       |
                RECOVERY EXECUTION
                       |
                   FINAL LLM


KEY POINT 1 — FAN-OUT
------------------------------------------------------------

Independent work can be executed in parallel.

        Employee MCP
        Product MCP

do not need to wait for each other.


KEY POINT 2 — FAN-IN
------------------------------------------------------------

Fan-in combines the results from parallel branches.

The state now contains information from both branches.


KEY POINT 3 — PARTIAL FAILURE
------------------------------------------------------------

One branch can fail while another succeeds.

Example:

    employee = SUCCESS
    product  = FAILED


KEY POINT 4 — STATE PRESERVATION
------------------------------------------------------------

The successful employee information remains in state.

We do NOT restart the entire workflow.


KEY POINT 5 — LLM RECOVERY ANALYSIS
------------------------------------------------------------

The LLM looks at the combined state and recommends:

    CONTINUE_PARTIALLY
    FALLBACK
    STOP


KEY POINT 6 — RECOVERY IS TARGETED
------------------------------------------------------------

If Product MCP failed:

    Recover Product

Do NOT unnecessarily repeat:

    Employee MCP


KEY POINT 7 — LLM DOES NOT DIRECTLY EXECUTE
------------------------------------------------------------

The LLM recommends the recovery.

The LangGraph workflow executes it.


KEY POINT 8 — DO NOT INVENT RESULTS
------------------------------------------------------------

A successful fallback execution does not automatically mean
the product is available.

The final LLM must only report what the state actually says.


============================================================
MENTAL MODEL
============================================================

                 LANGGRAPH
              = ORCHESTRATOR

                    |
          +---------+---------+
          |                   |
          v                   v
       MCP #1              MCP #2
       Employee             Product
          |                   |
       SUCCESS              FAILURE
          |                   |
          +---------+---------+
                    |
                  FAN-IN
                    |
                    v
              RECOVERY LOGIC
                    |
                    v
                FINAL LLM


This is the foundation of fault-tolerant enterprise
agent orchestration.

============================================================
"""