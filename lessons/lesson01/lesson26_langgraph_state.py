# ============================================================
# LESSON 26 — LANGGRAPH STATE
#
# Goal:
# Understand how State carries and accumulates information
# as it moves through multiple Nodes.
#
# RUNTIME ARCHITECTURE:
#
# START
#   ↓
# Node 1
#   ↓
# Node 2
#   ↓
# Node 3
#   ↓
# END
#
# State flows through every Node.
# Each Node can read existing State and update it.
# ============================================================


# ============================================================
# SETUP / DEFINITION
# 1. IMPORTS
# ============================================================

from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# ============================================================
# SETUP / DEFINITION
# 2. STATE
#
# State is the shared information carried through the graph.
#
# Think of it as a shared notebook that every Node can
# read from and update.
# ============================================================

class AgentState(TypedDict):

    user_name: str
    employee_id: int
    department: str


# ============================================================
# SETUP / DEFINITION
# 3. NODE 1 — IDENTIFY USER
#
# This Node adds the user's name to State.
# ============================================================

def identify_user(state: AgentState):

    print("\nNODE 1: Identifying user")

    return {
        "user_name": "Raj"
    }


# ============================================================
# SETUP / DEFINITION
# 4. NODE 2 — FIND EMPLOYEE ID
#
# This Node reads user_name from State and adds employee_id.
#
# State before:
#   user_name = Raj
#
# State after:
#   user_name = Raj
#   employee_id = 1
# ============================================================

def find_employee_id(state: AgentState):

    print("NODE 2: Finding employee ID")

    print(
        "Reading State → user_name:",
        state["user_name"]
    )

    return {
        "employee_id": 1
    }


# ============================================================
# SETUP / DEFINITION
# 5. NODE 3 — FIND DEPARTMENT
#
# This Node reads employee_id from State and adds department.
#
# State before:
#   user_name = Raj
#   employee_id = 1
#
# State after:
#   user_name = Raj
#   employee_id = 1
#   department = Engineering
# ============================================================

def find_department(state: AgentState):

    print("NODE 3: Finding department")

    print(
        "Reading State → employee_id:",
        state["employee_id"]
    )

    return {
        "department": "Engineering"
    }


# ============================================================
# SETUP / DEFINITION
# 6. CREATE THE GRAPH
# ============================================================

graph = StateGraph(AgentState)


# ============================================================
# SETUP / DEFINITION
# 7. ADD NODES
# ============================================================

graph.add_node(
    "identify_user",
    identify_user
)

graph.add_node(
    "find_employee_id",
    find_employee_id
)

graph.add_node(
    "find_department",
    find_department
)


# ============================================================
# SETUP / DEFINITION
# 8. ADD EDGES
#
# These Edges define the execution path:
#
# START
#   ↓
# identify_user
#   ↓
# find_employee_id
#   ↓
# find_department
#   ↓
# END
# ============================================================

graph.add_edge(
    START,
    "identify_user"
)

graph.add_edge(
    "identify_user",
    "find_employee_id"
)

graph.add_edge(
    "find_employee_id",
    "find_department"
)

graph.add_edge(
    "find_department",
    END
)


# ============================================================
# SETUP / DEFINITION
# 9. COMPILE GRAPH
# ============================================================

app = graph.compile()


# ============================================================
# RUNTIME FLOW
# 10. INITIAL STATE
#
# We start with empty information.
# ============================================================

initial_state = {
    "user_name": "",
    "employee_id": 0,
    "department": ""
}


# ============================================================
# RUNTIME FLOW
# 11. EXECUTE GRAPH
#
# State changes as it moves through the Nodes:
#
# {}
#   ↓
# {user_name: Raj}
#   ↓
# {user_name: Raj, employee_id: 1}
#   ↓
# {user_name: Raj, employee_id: 1,
#  department: Engineering}
# ============================================================

result = app.invoke(
    initial_state
)


# ============================================================
# RUNTIME FLOW
# 12. FINAL STATE
# ============================================================

print("\nFINAL STATE:")

print(result)


# ============================================================
# RUNTIME FLOW
# 13. FINAL INFORMATION
# ============================================================

print("\nEMPLOYEE INFORMATION:")

print(
    f"Name: {result['user_name']}"
)

print(
    f"Employee ID: {result['employee_id']}"
)

print(
    f"Department: {result['department']}"
)