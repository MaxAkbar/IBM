from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel
from typing_extensions import Literal

class FunctionParameters(BaseModel):
    type: str
    properties: Dict[str, Any]
    required: Optional[List[str]] = None

class Function(BaseModel):
    name: str
    description: str
    parameters: Optional[Dict[str, Any]] = None

class Tool(BaseModel):
    type: str = "function"
    function: Function  # Added this line to match OpenAI's tool format

class ToolFunction(BaseModel):
    name: str
    arguments: str

class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: ToolFunction

class MessageContent(BaseModel):
    type: str = "text"
    text: str

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[Union[str, List[MessageContent]]] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None