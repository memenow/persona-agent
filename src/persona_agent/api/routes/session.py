"""API routes for session management."""

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from typing import List, Dict, Any, Optional
import json
import asyncio

from src.persona_agent.api.models import (
    SessionResponse,
    SessionListResponse,
    CreateSessionRequest,
    SessionMessagesResponse,
    MessageResponse,
    SendMessageRequest,
    SendMessageResponse,
    SuccessResponse,
    ErrorResponse
)
from src.persona_agent.api.dependencies import get_agent_factory


router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get(
    "",
    response_model=SessionListResponse,
    summary="List all sessions",
    description="Returns a list of all active sessions."
)
async def list_sessions(
    agent_factory=Depends(get_agent_factory)
):
    """List all active sessions."""
    sessions = agent_factory.list_sessions()
    return SessionListResponse(sessions=[
        SessionResponse(
            id=session["id"],
            agent_id=session["agent_id"],
            persona_id=session["persona_id"],
            message_count=session["message_count"],
            created_at=session["created_at"],
            last_active=session["last_active"]
        )
        for session in sessions
    ])


@router.get(
    "/{session_id}",
    response_model=SessionResponse,
    summary="Get session details",
    description="Returns information about a specific session."
)
async def get_session(
    session_id: str,
    agent_factory=Depends(get_agent_factory)
):
    """Get information about a specific session."""
    session = agent_factory.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session with ID {session_id} not found")
    
    return SessionResponse(
        id=session.id,
        agent_id=session.agent_id,
        persona_id=session.persona_id,
        message_count=len(session.messages),
        created_at=session.created_at,
        last_active=session.last_active
    )


@router.post(
    "",
    response_model=SessionResponse,
    status_code=201,
    summary="Create a new session",
    description="Creates a new session with an agent."
)
async def create_session(
    request: CreateSessionRequest,
    agent_factory=Depends(get_agent_factory)
):
    """Create a new session with an agent."""
    # Check if the agent exists
    agent_info = agent_factory.get_agent(request.agent_id)
    if not agent_info:
        raise HTTPException(status_code=404, detail=f"Agent with ID {request.agent_id} not found")
    
    # Create the session
    session_id = agent_factory.create_session(request.agent_id)
    if not session_id:
        raise HTTPException(status_code=500, detail="Failed to create session")
    
    session = agent_factory.get_session(session_id)
    if not session:
        raise HTTPException(status_code=500, detail="Failed to retrieve created session")
    
    return SessionResponse(
        id=session.id,
        agent_id=session.agent_id,
        persona_id=session.persona_id,
        message_count=len(session.messages),
        created_at=session.created_at,
        last_active=session.last_active
    )


@router.delete(
    "/{session_id}",
    response_model=SuccessResponse,
    summary="Delete a session",
    description="Deletes a session with the specified ID."
)
async def delete_session(
    session_id: str,
    agent_factory=Depends(get_agent_factory)
):
    """Delete a session."""
    if not agent_factory.delete_session(session_id):
        raise HTTPException(status_code=404, detail=f"Session with ID {session_id} not found")
    
    return SuccessResponse(message=f"Session with ID {session_id} deleted successfully")


@router.get(
    "/{session_id}/messages",
    response_model=SessionMessagesResponse,
    summary="Get session messages",
    description="Returns all messages in a session."
)
async def get_session_messages(
    session_id: str,
    agent_factory=Depends(get_agent_factory)
):
    """Get all messages in a session."""
    session = agent_factory.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session with ID {session_id} not found")
    
    return SessionMessagesResponse(
        session_id=session.id,
        messages=[
            MessageResponse(
                role=message["role"],
                content=message["content"],
                timestamp=message["timestamp"]
            )
            for message in session.messages
        ]
    )


@router.post(
    "/{session_id}/messages",
    response_model=SendMessageResponse,
    summary="Send a message",
    description="Sends a message to the agent in the session and returns the response."
)
async def send_message(
    session_id: str,
    request: SendMessageRequest,
    agent_factory=Depends(get_agent_factory)
):
    """Send a message to the agent in the session."""
    session = agent_factory.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session with ID {session_id} not found")
    
    try:
        success, response = await agent_factory.send_message(session_id, request.message)
        
        if not success:
            # Check if this is a connection error and provide a friendly error message instead of a 500 status code
            if "Connection error" in response:
                # Return error message with 200 status code so client can handle it
                return SendMessageResponse(
                    session_id=session_id,
                    success=False,
                    response="OpenAI API Connection error. You can set ALLOW_LOCAL_FALLBACK=true environment variable to enable local response mode. Error details: " + response
                )
            else:
                # Other error messages, still return with 200 status code
                return SendMessageResponse(
                    session_id=session_id,
                    success=False,
                    response=response
                )
        
        return SendMessageResponse(
            session_id=session_id,
            success=success,
            response=response
        )
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        
        # Catch all exceptions and return a friendly error message instead of a 500 error
        return SendMessageResponse(
            session_id=session_id,
            success=False,
            response=f"Error processing message: {str(e)}",
            error_details=error_details
        )
