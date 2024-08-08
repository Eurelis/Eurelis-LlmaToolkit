from pydantic import BaseModel


class AgentHelloResponse(BaseModel):
    message: str
    is_active: bool
    is_search_active: bool
    is_similarity_active: bool
    max_history: int
    ui_params: dict


class AgentSearchResponse(BaseModel):
    id: float
    type: str
    url: str
    tags: str
    distance: float
    title: str
    baseline: str
    image: dict

class AgentSearchListResponse(BaseModel):
    content: list[AgentSearchResponse]
