from pydantic import BaseModel, Field
from src.agents.recommend import VenueRecommendations


class VenueRecommendationRequest(BaseModel):
    event_id: str = Field(description="The ID of the event", default="EVT-2026-028")
    top_n: int = Field(description="The number of recommendations to return", default=3)


class VenueRecommendationResponse(VenueRecommendations): ...


class IndexDocumentsRequest(BaseModel):
    event_history_path: str = Field(
        description="The path to the event history file",
        default="data/venue/event_history.json",
    )


class IndexDocumentsResponse(BaseModel):
    success: bool = Field(
        ..., description="Whether the documents were indexed successfully"
    )
    total_documents: int = Field(
        ..., description="The total number of documents indexed"
    )
