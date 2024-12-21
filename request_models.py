from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from typing_extensions import Literal
from models import Tool, ChatMessage

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="granite-3.0-8b-instruct")
    messages: List[ChatMessage]
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    max_completion_tokens: Optional[int] = Field(default=100, gt=0)
    stream: Optional[bool] = Field(default=False)