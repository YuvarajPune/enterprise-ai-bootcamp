
"""
Lesson 53.2 — Create an A2A Support Agent Server

Goal:
Expose the Support Agent through the A2A protocol.

High-level architecture:
A2A Client
    |
    | JSON-RPC request
    v
A2A Server (/a2a)
    |
    v
SupportAgentExecutor
    |
    v
Simple support-response function
    |
    v
A2A task result

Responsibilities:
- Agent Card: describes the agent and its capabilities.
- AgentExecutor: processes incoming tasks.
- TaskStore: stores task information in memory.
- JSON-RPC routes: expose the A2A communication endpoint.

This first version uses a simple Python response, not an LLM.
"""

import uvicorn

from a2a.helpers import (
    get_message_text,
    new_task_from_user_message,
    new_text_message,
    new_text_part,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
)
from a2a.types.a2a_pb2 import TaskState
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
from starlette.applications import Starlette


# --------------------------------------------------
# 1. SERVER CONFIGURATION
# --------------------------------------------------

HOST = "0.0.0.0"
PORT = 8002

# Used by clients running in the same Codespace.
AGENT_URL = f"http://127.0.0.1:{PORT}"
A2A_ENDPOINT = f"{AGENT_URL}/a2a"


# --------------------------------------------------
# 2. SUPPORT AGENT LOGIC
# --------------------------------------------------

def process_support_request(query: str) -> str:
    """
    Temporary agent logic.
    We will replace this with Gemini reasoning later.
    """

    if not query.strip():
        return "Please provide a customer support request."

    return (
        "Support Agent received your request: "
        f"{query}\n"
        "Status: Request received for analysis."
    )


# --------------------------------------------------
# 3. AGENT EXECUTOR
# --------------------------------------------------

class SupportAgentExecutor(AgentExecutor):
    """Receives and processes A2A tasks."""

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:

        # Create a task when this is a new request.
        if context.current_task:
            task = context.current_task
        else:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)

        # Manage the task lifecycle.
        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task.id,
            context_id=task.context_id,
        )

        await updater.update_status(
            state=TaskState.TASK_STATE_WORKING,
            message=new_text_message("Support Agent is processing the request."),
        )

        # Read the user's message and execute the agent logic.
        query = get_message_text(context.message)
        result = process_support_request(query or "")

        # Return the result as a task artifact.
        await updater.add_artifact(
            parts=[new_text_part(text=result, media_type="text/plain")]
        )

        # Mark the task as completed.
        await updater.update_status(
            state=TaskState.TASK_STATE_COMPLETED,
            message=new_text_message("Support task completed."),
        )

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        raise NotImplementedError("Task cancellation is not implemented.")


# --------------------------------------------------
# 4. AGENT CARD — DISCOVERY INFORMATION
# --------------------------------------------------

agent_card = AgentCard(
    name="Support Agent",
    description="Analyzes customer support requests.",
    version="1.0.0",
    supported_interfaces=[
        AgentInterface(
            url=A2A_ENDPOINT,
            protocol_binding="JSONRPC",
            protocol_version="1.0",
        )
    ],
    capabilities=AgentCapabilities(
        streaming=False,
    ),
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    skills=[
        AgentSkill(
            id="ticket_analysis",
            name="Ticket Analysis",
            description="Receives and processes customer support requests.",
            tags=["support", "customer-service", "ticket-analysis"],
            examples=[
                "Analyze customer complaints and identify recurring issues."
            ],
            input_modes=["text/plain"],
            output_modes=["text/plain"],
        )
    ],
)


# --------------------------------------------------
# 5. CONNECT EXECUTOR, TASK STORE AND A2A ROUTES
# --------------------------------------------------

request_handler = DefaultRequestHandler(
    agent_executor=SupportAgentExecutor(),
    task_store=InMemoryTaskStore(),
    agent_card=agent_card,
)

routes = []

# Publishes the Agent Card for discovery.
routes.extend(create_agent_card_routes(agent_card))

# Exposes the JSON-RPC communication endpoint.
routes.extend(create_jsonrpc_routes(request_handler, rpc_url="/a2a"))

app = Starlette(routes=routes)


# --------------------------------------------------
# 6. START THE SERVER
# --------------------------------------------------

if __name__ == "__main__":
    print("Starting Support Agent A2A Server...")
    print(f"Agent Card: {AGENT_URL}{AGENT_CARD_WELL_KNOWN_PATH}")
    print(f"A2A endpoint: {A2A_ENDPOINT}")

    uvicorn.run(app, host=HOST, port=PORT)
