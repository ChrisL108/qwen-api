from pydantic import BaseModel, Field
from typing import List, Optional, Union
from typing_extensions import Literal

# OpenAI API Models
class ImageURL(BaseModel):
    url: str

class ContentItem(BaseModel):
    type: Literal["text", "image_url"]
    text: Optional[str] = None
    image_url: Optional[ImageURL] = None

class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: Union[str, List[ContentItem]]

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 300
    temperature: Optional[float] = 1.0
    stream: Optional[bool] = False

class Delta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class Choice(BaseModel):
    index: int
    message: Optional[Message] = None
    delta: Optional[Delta] = None
    finish_reason: Optional[str] = None

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None

# Ollama API Models
class GenerateRequest(BaseModel):
    model: str = Field("age-estimation", description="Model name to use")
    prompt: str = Field(..., description="Prompt to generate a response for")
    images: Optional[List[str]] = Field(None, description="Base64 encoded images or URLs")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")
    options: Optional[dict] = Field(None, description="Additional model parameters")
