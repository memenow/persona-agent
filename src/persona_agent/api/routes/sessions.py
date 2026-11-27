"""REST routes for conversation sessions and messages."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from persona_agent.agents.service import AgentService
from persona_agent.api.dependencies import get_agent_service
from persona_agent.api.schemas import (
    CreateSessionRequest,
    MessageResponse,
    SendMessageRequest,
    SendMessageResponse,
    SessionListResponse,
    SessionMessagesResponse,
    SessionResponse,
    SuccessResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions", tags=["sessions"])

AgentServiceDep = Annotated[AgentService, Depends(get_agent_service)]


@router.get("", response_model=SessionListResponse)
async def list_sessions(agent_service: AgentServiceDep) -> SessionListResponse:
    return SessionListResponse(
        sessions=[
            SessionResponse(
                id=session.id,
                agent_id=session.agent_id,
                persona_id=session.persona_id,
                message_count=len(session.messages),
                created_at=session.created_at,
                last_active=session.last_active,
            )
            for session in agent_service.list_sessions()
        ]
    )


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str, agent_service: AgentServiceDep
) -> SessionResponse:
    session = agent_service.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=404, detail=f"Session with ID {session_id} not found"
        )
    return SessionResponse(
        id=session.id,
        agent_id=session.agent_id,
        persona_id=session.persona_id,
        message_count=len(session.messages),
        created_at=session.created_at,
        last_active=session.last_active,
    )


@router.post("", response_model=SessionResponse, status_code=201)
async def create_session(
    request: CreateSessionRequest, agent_service: AgentServiceDep
) -> SessionResponse:
    session = agent_service.create_session(request.agent_id)
    if session is None:
        raise HTTPException(
            status_code=404, detail=f"Agent with ID {request.agent_id} not found"
        )
    return SessionResponse(
        id=session.id,
        agent_id=session.agent_id,
        persona_id=session.persona_id,
        message_count=len(session.messages),
        created_at=session.created_at,
        last_active=session.last_active,
    )


@router.delete("/{session_id}", response_model=SuccessResponse)
async def delete_session(
    session_id: str, agent_service: AgentServiceDep
) -> SuccessResponse:
    if not agent_service.delete_session(session_id):
        raise HTTPException(
            status_code=404, detail=f"Session with ID {session_id} not found"
        )
    return SuccessResponse(message=f"Session with ID {session_id} deleted successfully")


@router.get("/{session_id}/messages", response_model=SessionMessagesResponse)
async def get_session_messages(
    session_id: str, agent_service: AgentServiceDep
) -> SessionMessagesResponse:
    session = agent_service.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=404, detail=f"Session with ID {session_id} not found"
        )
    return SessionMessagesResponse(
        session_id=session.id,
        messages=[
            MessageResponse(
                role=message["role"],
                content=message["content"],
                timestamp=message["timestamp"],
            )
            for message in session.messages
        ],
    )


@router.post("/{session_id}/messages", response_model=SendMessageResponse)
async def send_message(
    session_id: str,
    request: SendMessageRequest,
    agent_service: AgentServiceDep,
) -> SendMessageResponse:
    if agent_service.get_session(session_id) is None:
        raise HTTPException(
            status_code=404, detail=f"Session with ID {session_id} not found"
        )

    try:
        success, response = await agent_service.send_message(
            session_id, request.message
        )
        return SendMessageResponse(
            session_id=session_id,
            success=success,
            response=response,
        )
    except Exception:
        logger.exception("Error sending message in session %s", session_id)
        return SendMessageResponse(
            session_id=session_id,
            success=False,
            response="Error processing message",
        )
