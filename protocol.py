import time
from typing import Optional, List, Dict, Any, Union

from pydantic import BaseModel, Field


# class ModelPermission(BaseModel):
#     id: str = Field(default_factory=lambda: f"modelperm-{secrets.token_hex(12)}")
#     object: str = "model_permission"
#     created: int = Field(default_factory=lambda: int(time.time()))
#     allow_create_engine: bool = False
#     allow_sampling: bool = True
#     allow_logprobs: bool = True
#     allow_search_indices: bool = True
#     allow_view: bool = True
#     allow_fine_tuning: bool = False
#     organization: str = "*"
#     group: Optional[str] = None
#     is_blocking: str = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "fhxk-iao"
    root: Optional[str] = None
    parent: Optional[str] = None
    # permission: List[ModelPermission] = []


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class EmbeddingsRequest(BaseModel):
    model: Optional[str] = None
    engine: Optional[str] = None
    input: Union[str, List[Any]]
    user: Optional[str] = None
    encoding_format: Optional[str] = None


class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]
    model: str
    usage: UsageInfo