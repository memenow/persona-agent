"""Pydantic request and response schemas for the REST API."""

from typing import Any

from pydantic import BaseModel, Field


class PersonaResponse(BaseModel):
    id: str
    name: str
    description: str


class PersonaListResponse(BaseModel):
    personas: list[PersonaResponse]


class PersonaDetailResponse(PersonaResponse):
    personal_background: dict[str, Any] = Field(default_factory=dict)
    language_style: dict[str, Any] = Field(default_factory=dict)
    knowledge_domains: dict[str, Any] = Field(default_factory=dict)
    interaction_samples: list[dict[str, Any]] = Field(default_factory=list)


class CreatePersonaRequest(BaseModel):
    name: str
    description: str = ""
    personal_background: dict[str, Any] | None = None
    language_style: dict[str, Any] | None = None
    knowledge_domains: dict[str, Any] | None = None
    interaction_samples: list[dict[str, Any]] | None = None
    system_prompt: str | None = None


class UpdatePersonaRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    personal_background: dict[str, Any] | None = None
    language_style: dict[str, Any] | None = None
    knowledge_domains: dict[str, Any] | None = None
    interaction_samples: list[dict[str, Any]] | None = None
    system_prompt: str | None = None


class AgentResponse(BaseModel):
    id: str
    persona_id: str
    name: str
    created_at: float


class AgentListResponse(BaseModel):
    agents: list[AgentResponse]


class CreateAgentRequest(BaseModel):
    persona_id: str
    model: str | None = None


class SessionResponse(BaseModel):
    id: str
    agent_id: str
    persona_id: str
    message_count: int
    created_at: float
    last_active: float


class SessionListResponse(BaseModel):
    sessions: list[SessionResponse]


class CreateSessionRequest(BaseModel):
    agent_id: str


class MessageResponse(BaseModel):
    role: str
    content: str
    timestamp: float


class SessionMessagesResponse(BaseModel):
    session_id: str
    messages: list[MessageResponse]


class SendMessageRequest(BaseModel):
    message: str


class SendMessageResponse(BaseModel):
    session_id: str
    success: bool
    response: str
    error_details: str | None = None


class SuccessResponse(BaseModel):
    success: bool = True
    message: str
