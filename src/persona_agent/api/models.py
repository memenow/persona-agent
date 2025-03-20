"""Data models for API requests and responses."""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


# Persona API Models
class PersonaResponse(BaseModel):
    """Response model for persona operations."""
    id: str = Field(..., description="Unique ID for the persona")
    name: str = Field(..., description="Name of the persona")
    description: str = Field(..., description="Description of the persona")


class PersonaListResponse(BaseModel):
    """Response model for listing personas."""
    personas: List[PersonaResponse] = Field(..., description="List of available personas")


class PersonaDetailResponse(PersonaResponse):
    """Detailed persona information response."""
    personal_background: Dict[str, Any] = Field(default_factory=dict, description="Personal background information")
    language_style: Dict[str, Any] = Field(default_factory=dict, description="Language style characteristics")
    knowledge_domains: Dict[str, Any] = Field(default_factory=dict, description="Knowledge domains and expertise")
    interaction_samples: List[Dict[str, Any]] = Field(default_factory=list, description="Sample interactions")


class CreatePersonaRequest(BaseModel):
    """Request model for creating a persona."""
    name: str = Field(..., description="Name of the persona")
    description: str = Field(default="", description="Description of the persona")
    personal_background: Optional[Dict[str, Any]] = Field(default=None, description="Personal background information")
    language_style: Optional[Dict[str, Any]] = Field(default=None, description="Language style characteristics")
    knowledge_domains: Optional[Dict[str, Any]] = Field(default=None, description="Knowledge domains and expertise")
    interaction_samples: Optional[List[Dict[str, Any]]] = Field(default=None, description="Sample interactions")
    system_prompt: Optional[str] = Field(default=None, description="Custom system prompt to use")


class UpdatePersonaRequest(BaseModel):
    """Request model for updating a persona."""
    name: Optional[str] = Field(default=None, description="Name of the persona")
    description: Optional[str] = Field(default=None, description="Description of the persona")
    personal_background: Optional[Dict[str, Any]] = Field(default=None, description="Personal background information")
    language_style: Optional[Dict[str, Any]] = Field(default=None, description="Language style characteristics")
    knowledge_domains: Optional[Dict[str, Any]] = Field(default=None, description="Knowledge domains and expertise")
    interaction_samples: Optional[List[Dict[str, Any]]] = Field(default=None, description="Sample interactions")
    system_prompt: Optional[str] = Field(default=None, description="Custom system prompt to use")


# Agent API Models
class AgentResponse(BaseModel):
    """Response model for agent operations."""
    id: str = Field(..., description="Unique ID for the agent")
    persona_id: str = Field(..., description="ID of the persona used by this agent")
    name: str = Field(..., description="Name of the agent")
    created_at: float = Field(..., description="Timestamp when the agent was created")


class AgentListResponse(BaseModel):
    """Response model for listing agents."""
    agents: List[AgentResponse] = Field(..., description="List of available agents")


class CreateAgentRequest(BaseModel):
    """Request model for creating an agent."""
    persona_id: str = Field(..., description="ID of the persona to use")
    model: Optional[str] = Field(default=None, description="Model to use for the agent")


# Session API Models
class SessionResponse(BaseModel):
    """Response model for session operations."""
    id: str = Field(..., description="Unique ID for the session")
    agent_id: str = Field(..., description="ID of the agent for this session")
    persona_id: str = Field(..., description="ID of the persona for this session")
    message_count: int = Field(..., description="Number of messages in the session")
    created_at: float = Field(..., description="Timestamp when the session was created")
    last_active: float = Field(..., description="Timestamp when the session was last active")


class SessionListResponse(BaseModel):
    """Response model for listing sessions."""
    sessions: List[SessionResponse] = Field(..., description="List of available sessions")


class CreateSessionRequest(BaseModel):
    """Request model for creating a session."""
    agent_id: str = Field(..., description="ID of the agent to use for this session")


class MessageResponse(BaseModel):
    """Response model for session messages."""
    role: str = Field(..., description="Role of the message sender (user or assistant)")
    content: str = Field(..., description="Content of the message")
    timestamp: float = Field(..., description="Timestamp when the message was sent")


class SessionMessagesResponse(BaseModel):
    """Response model for listing session messages."""
    session_id: str = Field(..., description="ID of the session")
    messages: List[MessageResponse] = Field(..., description="List of messages in the session")


class SendMessageRequest(BaseModel):
    """Request model for sending a message to a session."""
    message: str = Field(..., description="Message to send")


class SendMessageResponse(BaseModel):
    """Response model for sending a message to a session."""
    session_id: str = Field(..., description="ID of the session")
    success: bool = Field(..., description="Whether the message was sent successfully")
    response: str = Field(..., description="Response from the agent")
    error_details: Optional[str] = Field(default=None, description="Detailed error information if available")


# Tool API Models
class ToolResponse(BaseModel):
    """Response model for tool operations."""
    id: str = Field(..., description="Tool identifier")
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    service_name: str = Field(..., description="Name of the service providing this tool")


class ToolListResponse(BaseModel):
    """Response model for listing tools."""
    tools: List[ToolResponse] = Field(..., description="List of available tools")


class ToolExecuteRequest(BaseModel):
    """Request model for executing a tool."""
    tool_id: str = Field(..., description="ID of the tool to execute")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the tool")
    session_id: Optional[str] = Field(default=None, description="Optional session ID to associate with this tool execution")


class ToolExecuteResponse(BaseModel):
    """Response model for tool execution."""
    tool_id: str = Field(..., description="ID of the tool that was executed")
    success: bool = Field(..., description="Whether the tool execution was successful")
    result: Any = Field(..., description="Result of the tool execution")
    execution_time: float = Field(..., description="Time taken to execute the tool in seconds")


class ServiceResponse(BaseModel):
    """Response model for service operations."""
    name: str = Field(..., description="Service name")
    description: str = Field(..., description="Service description")
    available: bool = Field(..., description="Whether the service is available")
    tool_count: int = Field(..., description="Number of tools provided by this service")


class ServiceListResponse(BaseModel):
    """Response model for listing services."""
    services: List[ServiceResponse] = Field(..., description="List of available services")


# Common API Models
class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")


class SuccessResponse(BaseModel):
    """Standard success response."""
    success: bool = Field(default=True, description="Success indicator")
    message: str = Field(..., description="Success message") 