"""API routes for agent management."""

from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any, Optional

from src.persona_agent.api.models import (
    AgentResponse,
    AgentListResponse,
    CreateAgentRequest,
    SuccessResponse,
    ErrorResponse
)
from src.persona_agent.api.dependencies import get_agent_factory, get_persona_manager


router = APIRouter(prefix="/agents", tags=["agents"])


@router.get(
    "",
    response_model=AgentListResponse,
    summary="List all agents",
    description="Returns a list of all available agents."
)
async def list_agents(
    agent_factory=Depends(get_agent_factory)
):
    """List all available agents."""
    agents = agent_factory.list_agents()
    return AgentListResponse(agents=[
        AgentResponse(
            id=agent["id"],
            persona_id=agent["persona_id"],
            name=agent["name"],
            created_at=agent["created_at"]
        )
        for agent in agents
    ])


@router.get(
    "/{agent_id}",
    response_model=AgentResponse,
    summary="Get agent details",
    description="Returns information about a specific agent."
)
async def get_agent(
    agent_id: str,
    agent_factory=Depends(get_agent_factory)
):
    """Get information about a specific agent."""
    agent_info = agent_factory.get_agent(agent_id)
    if not agent_info:
        raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
    
    return AgentResponse(
        id=agent_info["id"],
        persona_id=agent_info["persona_id"],
        name=agent_info["agent"].name,
        created_at=agent_info["created_at"]
    )


@router.post(
    "",
    response_model=AgentResponse,
    status_code=201,
    summary="Create a new agent",
    description="Creates a new agent based on a persona."
)
async def create_agent(
    request: CreateAgentRequest,
    agent_factory=Depends(get_agent_factory),
    persona_manager=Depends(get_persona_manager)
):
    """Create a new agent based on a persona."""
    # Get the persona
    persona = persona_manager.get_persona(request.persona_id)
    if not persona:
        raise HTTPException(status_code=404, detail=f"Persona with ID {request.persona_id} not found")
    
    # Create the agent
    try:
        model = request.model or None
        agent_id = await agent_factory.create_agent(persona, model=model)
        
        # Get the created agent
        agent_info = agent_factory.get_agent(agent_id)
        if not agent_info:
            raise HTTPException(status_code=500, detail="Failed to create agent")
        
        return AgentResponse(
            id=agent_info["id"],
            persona_id=agent_info["persona_id"],
            name=agent_info["agent"].name,
            created_at=agent_info["created_at"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating agent: {str(e)}")


@router.delete(
    "/{agent_id}",
    response_model=SuccessResponse,
    summary="Delete an agent",
    description="Deletes an agent with the specified ID."
)
async def delete_agent(
    agent_id: str,
    agent_factory=Depends(get_agent_factory)
):
    """Delete an agent."""
    if not agent_factory.delete_agent(agent_id):
        raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
    
    return SuccessResponse(message=f"Agent with ID {agent_id} deleted successfully") 