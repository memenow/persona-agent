"""Public A2A discovery routes for aggregate agent cards and persona listing."""

from typing import Annotated

from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from persona_agent.a2a.registry import A2ARegistry
from persona_agent.api.dependencies import get_a2a_registry

router = APIRouter(tags=["a2a"])

A2ARegistryDep = Annotated[A2ARegistry, Depends(get_a2a_registry)]


@router.get(AGENT_CARD_WELL_KNOWN_PATH)
@router.get("/.well-known/agent.json")
async def get_aggregate_agent_card(registry: A2ARegistryDep) -> JSONResponse:
    """Serve the aggregate A2A v1.0 AgentCard at canonical and legacy well-known paths."""
    return JSONResponse(content=registry.build_aggregate_card())


@router.get("/a2a/personas")
async def list_a2a_personas(registry: A2ARegistryDep) -> JSONResponse:
    """Return summary metadata for all mounted A2A persona sub-apps."""
    return JSONResponse(content={"agents": registry.list_personas()})
