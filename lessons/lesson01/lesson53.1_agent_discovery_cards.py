
"""
LESSON 53.1 — A2A: AGENT DISCOVERY AND AGENT CARDS

GOAL
Use an LLM to interpret a request and identify a required capability.
Use Python to validate that capability and select a matching Agent Card.

HIGH-LEVEL ARCHITECTURE

    User Request
         |
         v
    Coordinator LLM (Gemini)
    Identifies the required capability
         |
         v
    Python Validation
    Verifies the capability is in the allowed list
         |
         v
    Local Agent Registry
    Searches Agent Cards for an exact skill match
         |
         v
    Selected Agent Card
    Displays the selected agent and its endpoint

COMPONENT RESPONSIBILITIES

1. Gemini LLM:
   Interprets the user's request and identifies a capability.

2. Agent Cards:
   Declare agent identity, skills and endpoint.

3. Python application:
   Validates the LLM response and selects a matching agent.

4. A2A endpoint:
   Illustrates where a remote agent could be contacted.
   This lesson does NOT make an actual A2A network call.

SETUP
- Set GEMINI_API_KEY in your .env file.
- Install google-genai and python-dotenv if needed.
- Run:
  python lessons/lesson01/lesson53.1_agent_discovery_cards.py
"""

# ============================================================
# SECTION 1: IMPORTS AND GEMINI CLIENT SETUP
# Purpose: Load environment variables and initialize the LLM.
# ============================================================

import os
import time

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Check your .env file."
    )

client = genai.Client(api_key=api_key)


# ============================================================
# SECTION 2: LOCAL AGENT REGISTRY
# Purpose: Define the agents available to this application.
#
# These are simplified teaching examples, not complete,
# protocol-compliant A2A Agent Cards.
# ============================================================


# ============================================================
# SECTION 2: LOCAL AGENT REGISTRY
# Purpose: Store richer A2A-style Agent Card metadata.
#
# These are educational examples, not verified live endpoints.
# ============================================================

agent_cards = [
    {
        "name": "Finance Agent",
        "description": "Analyzes financial performance and risk",
        "version": "1.0.0",
        "supportedInterfaces": [
            {
                "url": "http://localhost:8001/a2a",
                "protocolBinding": "JSONRPC",
                "protocolVersion": "1.0"
            }
        ],
        "capabilities": {
            "streaming": True,
            "pushNotifications": False
        },
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [
            "financial_analysis",
            "risk_analysis"
        ]
    },
    {
        "name": "Support Agent",
        "description": "Analyzes customer issues and support requests",
        "version": "1.0.0",
        "supportedInterfaces": [
            {
                "url": "http://localhost:8002/a2a",
                "protocolBinding": "JSONRPC",
                "protocolVersion": "1.0"
            }
        ],
        "capabilities": {
            "streaming": True,
            "pushNotifications": False
        },
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [
            "customer_support",
            "ticket_analysis"
        ]
    }
]



# ============================================================
# SECTION 3: DISPLAY REGISTERED AGENTS
# Purpose: Inspect available agents and their declared skills.
# ============================================================

print("\n========== AGENT REGISTRY ==========")

for card in agent_cards:
    print(f"\nAgent       : {card['name']}")
    print(f"Description : {card['description']}")
    print(f"Skills      : {', '.join(card['skills'])}")
    print(f"Endpoint    : {card['supportedInterfaces'][0]['url']}")


# ============================================================
# SECTION 4: USER REQUEST
# Purpose: Define the task that needs a suitable agent.
# In a real application, this may come from a chatbot or API.
# ============================================================

user_request = (
    "Analyze customer complaints and identify recurring "
    "support issues."
)

print("\n========== USER REQUEST ==========")
print(user_request)


# ============================================================
# SECTION 5: PREPARE ALLOWED CAPABILITIES
# Purpose: Derive the allowed skills from Agent Cards.
#
# This avoids maintaining a separate list that could become
# inconsistent with the registry.
# ============================================================

allowed_capabilities = sorted({
    skill
    for card in agent_cards
    for skill in card["skills"]
})

capability_list = "\n".join(
    f"- {skill}" for skill in allowed_capabilities
)


# ============================================================
# SECTION 6: LLM-BASED CAPABILITY IDENTIFICATION
# Purpose: Ask Gemini to identify the most relevant skill.
#
# The LLM recommends a capability. It does not select an
# endpoint or execute a remote agent.
# ============================================================

prompt = f"""
You identify the capability required for a user request.

Choose exactly one capability from this list:
{capability_list}

Rules:
- Return one capability only.
- Preserve the spelling exactly.
- Do not invent capabilities.
- Do not add explanations or formatting.

User request:
{user_request}
"""

print("\n========== LLM PROCESSING ==========")

# Retry temporary 503 service-unavailable errors.
# Wait 2 seconds, then 4 seconds, for a maximum of 3 attempts.
response = None

for attempt in range(3):
    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )
        break

    except Exception as error:
        error_message = str(error)

        # Retry only temporary service-unavailable responses.
        if (
            "503" not in error_message
            and "UNAVAILABLE" not in error_message
        ):
            raise

        if attempt == 2:
            print("Gemini remained unavailable after 3 attempts.")
            raise

        wait_seconds = 2 ** (attempt + 1)

        print(
            f"Gemini temporarily unavailable. "
            f"Retrying in {wait_seconds} seconds..."
        )

        time.sleep(wait_seconds)

# Check that the LLM returned usable text.
if response is None or not response.text:
    raise RuntimeError("Gemini returned an empty response.")

required_capability = response.text.strip().strip("`").strip()

print("LLM-selected capability:", required_capability)


# ============================================================
# SECTION 7: VALIDATE THE LLM RESPONSE
# Purpose: Reject any capability that is not in the allowed list.
#
# This is an application-side control: the LLM output is
# treated as a recommendation, not as trusted application data.
# ============================================================

print("\n========== CAPABILITY VALIDATION ==========")

if required_capability not in allowed_capabilities:
    print("Validation failed.")
    print("Returned value:", repr(required_capability))
    print("Allowed values:", allowed_capabilities)

    raise SystemExit(
        "No agent selected because the capability was invalid."
    )

print("Validation successful.")


# ============================================================
# SECTION 8: AGENT DISCOVERY AND CAPABILITY MATCHING
# Purpose: Find Agent Cards that declare the required skill.
#
# Python performs deterministic exact matching.
# ============================================================

print("\n========== AGENT DISCOVERY ==========")

matched_agents = [
    card
    for card in agent_cards
    if required_capability in card["skills"]
]

if not matched_agents:
    print("No registered agent supports this capability.")
    raise SystemExit("Agent discovery could not find a match.")

# For this experiment, select the first exact match.
# Production systems may also check permissions, availability,
# agent health and ranking when multiple agents match.
selected_agent = matched_agents[0]



# ============================================================
# SECTION 9: DISPLAY THE SELECTED AGENT
# Purpose: Display the agent selected through capability matching.
#
# The endpoint is now nested inside supportedInterfaces.
# This section displays the endpoint but does NOT contact it.
# ============================================================

print("\n========== SELECTED AGENT ==========")

print("Agent        :", selected_agent["name"])
print("Description  :", selected_agent["description"])
print("Matched skill:", required_capability)
print(
    "Endpoint     :",
    selected_agent["supportedInterfaces"][0]["url"]
)
print(
    "Protocol     :",
    selected_agent["supportedInterfaces"][0]["protocolBinding"]
)
print(
    "Protocol version:",
    selected_agent["supportedInterfaces"][0]["protocolVersion"]
)

print("\nDiscovery completed successfully.")
print("No remote A2A task has been sent in this lesson.")
