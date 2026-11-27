"""REST routes for agent lifecycle management."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from persona_agent.agents.service import AgentService
from persona_agent.api.dependencies import get_agent_service, get_persona_repository
from persona_agent.api.schemas import (
    AgentListResponse,
    AgentResponse,
    CreateAgentRequest,
    SuccessResponse,
)
from persona_agent.personas.repository import PersonaRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])

AgentServiceDep = Annotated[AgentService, Depends(get_agent_service)]
PersonaRepoDep = Annotated[PersonaRepository, Depends(get_persona_repository)]


@router.get("", response_model=AgentListResponse)
async def list_agents(agent_service: AgentServiceDep) -> AgentListResponse:
    return AgentListResponse(
        agents=[
            AgentResponse(
                id=record.id,
                persona_id=record.persona_id,
                name=record.name,
                created_at=record.created_at,
            )
            for record in agent_service.list_agents()
        ]
    )


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: str, agent_service: AgentServiceDep) -> AgentResponse:
    record = agent_service.get_agent(agent_id)
    if record is None:
        raise HTTPException(
            status_code=404, detail=f"Agent with ID {agent_id} not found"
        )
    return AgentResponse(
        id=record.id,
        persona_id=record.persona_id,
        name=record.name,
        created_at=record.created_at,
    )


@router.post("", response_model=AgentResponse, status_code=201)
async def create_agent(
    request: CreateAgentRequest,
    agent_service: AgentServiceDep,
    persona_repo: PersonaRepoDep,
) -> AgentResponse:
    persona = persona_repo.get_persona(request.persona_id)
    if persona is None:
        raise HTTPException(
            status_code=404, detail=f"Persona with ID {request.persona_id} not found"
        )

    try:
        record = await agent_service.create_agent(persona, model=request.model)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error creating agent for persona %s", request.persona_id)
        raise HTTPException(status_code=500, detail="Error creating agent") from exc

    return AgentResponse(
        id=record.id,
        persona_id=record.persona_id,
        name=record.name,
        created_at=record.created_at,
    )


@router.delete("/{agent_id}", response_model=SuccessResponse)
async def delete_agent(
    agent_id: str, agent_service: AgentServiceDep
) -> SuccessResponse:
    if not agent_service.delete_agent(agent_id):
        raise HTTPException(
            status_code=404, detail=f"Agent with ID {agent_id} not found"
        )
    return SuccessResponse(message=f"Agent with ID {agent_id} deleted successfully")
