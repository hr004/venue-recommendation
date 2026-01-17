import json
import logging
from fastapi import APIRouter, Depends, status
from src.orchestrator import VenueRecommendationOrchestrator
from src.datamodel.api import (
    IndexDocumentsRequest,
    IndexDocumentsResponse,
    VenueRecommendationRequest,
    VenueRecommendationResponse,
)
from src.engine.vector_store import OpenSearchEngine
from src.fastapi_app.http_exception import UnprocessableEntityHTTPError

logger = logging.getLogger(__name__)


def get_current_requests_db():
    current_requests_db = json.load(open("data/venue/current_requests.json"))
    return {event["event_id"]: event for event in current_requests_db}


file_router = APIRouter(
    prefix="/api/v1",
)


@file_router.post(
    "/venues/recommend",
    status_code=200,
    response_model=VenueRecommendationResponse,
)
async def get_venue_recommendations(
    request: VenueRecommendationRequest,
    current_requests_db: dict = Depends(get_current_requests_db),
) -> VenueRecommendationResponse:
    logger.debug("Search nearest venue recommendations")
    if request.event_id not in current_requests_db:
        raise UnprocessableEntityHTTPError(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Event not in current requests database",
        )
    event = current_requests_db[request.event_id]
    vector_engine = OpenSearchEngine()
    retrieved_documents = await vector_engine.search_documents(event)
    venue_recommendation_agent = VenueRecommendationOrchestrator()
    recommendations = await venue_recommendation_agent.recommend(
        event, retrieved_documents
    )
    return recommendations


@file_router.post(
    "/index",
    status_code=status.HTTP_201_CREATED,
    response_model=IndexDocumentsResponse,
)
async def index_documents(
    request: IndexDocumentsRequest,
) -> IndexDocumentsResponse:
    logger.debug("Index documents")
    vector_engine = OpenSearchEngine()
    success, total_documents = await vector_engine.index_documents(
        request.event_history_path,
    )
    return IndexDocumentsResponse(
        success=success,
        total_documents=total_documents,
    )
