import json
import logging
import os
import pathlib
from datetime import datetime
from typing import Any, Dict
from src.config import config
from src.utils import read_json
from langchain_community.vectorstores.opensearch_vector_search import OpenSearchVectorSearch
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


# from langchain.embeddings import CacheBackedEmbeddings
# from langchain.storage import LocalFileStore

ROOT = pathlib.Path(__file__).parent.parent.parent
CLIENT_PROFILE_PATH = ROOT / "data" / "venue" / "client_profiles.json"
VENUE_PATH = ROOT / "data" / "venue" / "venues.json"

with open(CLIENT_PROFILE_PATH) as f:
    CLIENT_PROFILE = {client["client_id"]: client for client in json.load(f)}

with open(VENUE_PATH) as f:
    VENUE = {venue["venue_id"]: venue for venue in json.load(f)}


def create_event_embedding_text(event: Dict[str, Any]) -> str:
    """
    Create composite text for embedding that captures event-venue match pattern.
    This text will be vectorized for semantic similarity search.
    """
    # Extract arrays as comma-separated strings
    key_requirements = ", ".join(event.get("key_requirements", []))
    success_factors = ", ".join(event.get("success_factors", []))
    challenges = ", ".join(event.get("challenges", []))
    positive_feedback = "; ".join(event.get("positive_feedback", []))
    negative_feedback = "; ".join(event.get("negative_feedback", []))
    notes = event.get("notes", "")

    # Build comprehensive embedding text
    text = f"""
        Event ID: {event.get('event_id', '')}
        Event Name: {event.get('event_name', '')}
        Event Type: {event.get('event_type', '')}
        Attendee Count: {event.get('attendee_count', '')}
        Client Name: {event.get('client_name', '')}
        Venue ID: {event.get('venue_id', '')}
        Venue Name: {event.get('venue_name', '')}
        Venue City: {event.get('city', '')}
        Key Requirements: {key_requirements}
        Requirements Met: {event.get('requirements_met', '')}
        Success Factors: {success_factors}
        Challenges: {challenges}
        Positive Feedback: {positive_feedback}
        Negative Feedback: {negative_feedback}
        Notes: {notes}
        Outcome: {event.get('outcome', '')}
    """
    return text


def create_document_metadata(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create metadata dictionary matching the mappings.json structure.
    This enables structured filtering and retrieval.
    """
    # NOTE: some of the client profiles are missing
    metadata = {
        "event": get_event_metadata(event),
        "client": CLIENT_PROFILE.get(event.get("client_id")),
        "venue": VENUE.get(event.get("venue_id")),
    }
    return metadata


def get_event_metadata(event: Dict[str, Any]) -> Dict[str, Any]:
    event_metadata = {
        "event_id": event.get("event_id"),
        "event_name": event.get("event_name"),
        "event_type": event.get("event_type"),
        "client_name": event.get("client_name"),
        "key_requirements": event.get("key_requirements"),
        "success_factors": event.get("success_factors"),
        "challenges": event.get("challenges"),
        "duration_days": event.get("duration_days"),
        "duration_days": event.get("duration_days"),
        "event_dates": event.get("event_dates"),
        "venue_id": event.get("venue_id"),
        "city": event.get("city"),
        "total_cost": event.get("total_cost"),
        "venue_cost": event.get("venue_cost"),
        "other_costs": event.get("other_costs"),
        "budget_met": event.get("budget_met"),
        "event_style": event.get("event_style"),
        "requirements_met": event.get("requirements_met"),
        "client_rating": event.get("client_rating"),
        "venue_rating": event.get("venue_rating"),
        "average_rating": event.get("average_rating"),
        "overall_satisfaction": event.get("overall_satisfaction"),
        "would_recommend": event.get("would_recommend"),
        "would_rebook": event.get("would_rebook"),
        "rebooking_likelihood": event.get("rebooking_likelihood"),
        "indexed_at": datetime.now().isoformat(),
    }

    # Remove None values to avoid indexing issues
    return {k: v for k, v in event_metadata.items() if v is not None}


class OpenSearchEngine:
    def __init__(self):
        super().__init__()
        self.config_opensearch = {
            "embedding_function": self.get_embedding_client(),
            "engine": "faiss",
        }
        self.vectorstore = OpenSearchVectorSearch(
            opensearch_url=config.opensearch.url,
            index_name=config.opensearch.index_name,
            **self.config_opensearch,
        )

    def get_embedding_client(self):

        # _embedding_client = MockEmbeddings()
        _embedding_client = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            chunk_size=2000,
        )
        return _embedding_client

    async def __write_documents(self, documents, url, index_name):
        config = {
            "engine": "faiss",
            "ef_construction": 256,
            "bulk_size": len(documents),
            "m": 48,
        }
        try:
            _ = await OpenSearchVectorSearch.afrom_documents(
                documents,
                self.get_embedding_client(),
                opensearch_url=url,
                index_name=index_name,
                **config,
            )
        except Exception as e:
            logging.error(f"Error indexing documents: {e}")
            raise e

    def _build_filter_clause(self, event: Dict[str, Any]) -> str:
        filter_clauses = []

        attendee_count = event.get("attendee_count")
        if attendee_count:
            # allow ±20% range or minimum ±50 attendees for flexibility
            tolerance = max(int(attendee_count * 0.2), 50)
            min_attendees = max(0, attendee_count - tolerance)
            max_attendees = attendee_count + tolerance

            filter_clauses.append(
                {
                    "range": {
                        "metadata.venue.max_capacity": {
                            "gte": min_attendees,
                            "lte": max_attendees,
                        }
                    }
                }
            )

        # filter by location requirements cities (keyword search)
        location_requirements = event.get("location_requirements", {})
        cities = location_requirements.get("cities", [])
        if cities:
            city_names = []
            for city in cities:
                city_name = city.split(",")[0].strip() if "," in city else city.strip()
                city_names.append(city_name)

            filter_clauses.append(
                {
                    "bool": {
                        "should": [
                            {"prefix": {"metadata.venue.city.keyword": city_name}}
                            for city_name in city_names
                        ],
                        "minimum_should_match": 1,
                    }
                }
            )
        if filter_clauses:
            filter_clause = {"bool": {"should": filter_clauses}}
        return filter_clause

    async def __search_documents(
        self,
        query: str,
        k: int,
        event: Dict[str, Any],
        vectorstore: OpenSearchVectorSearch,
    ) -> list[Document]:
        filter_clause = self._build_filter_clause(event)
        try:
            retrieved_documents = await vectorstore.asimilarity_search(
                query=query,
                k=k,
                search_type="approximate_search",
                efficient_filter=filter_clause if filter_clause is not None else None,
            )

        except Exception as e:
            logging.error(f"Error searching documents: {e}")
            retrieved_documents = []
        return retrieved_documents



    async def index_documents(self, event_history_path: str):
        documents = []

        event_history = read_json(event_history_path)
        for event in event_history:
            documents.append(
                Document(
                    page_content=create_event_embedding_text(event),
                    metadata=create_document_metadata(event),
                )
            )

        await self.__write_documents(
            documents, config.opensearch.url, config.opensearch.index_name
        )
        return True, len(documents)

    def _build_query(self, event: Dict[str, Any]) -> str:
        query = f"""
        Event ID: {event.get('event_id', '')}
        Event Name: {event.get('event_name', '')}
        Event Type: {event.get('event_type', '')}
        Attendee Count: {event.get('attendee_count', '')}
        Client Name: {event.get('client_name', '')}
        Location : {event.get('location_preference', '')}
        Location Requirements: {event.get('location_requirements', [])}
        Key Requirements: {event.get('key_requirements', [])}
        Required Amenities: {event.get('required_amenities', [])}
        Preferred Amenities: {event.get('preferred_amenities', [])}
        Event Style: {event.get('event_style', '')}
        Special Requirements: {event.get('special_requirements', [])}
        Client Preferences: {event.get('client_preferences', [])}
        """
        return query

    # TODO: add filter and k in the request API
    async def search_documents(self, event, k=10):
        # build query from event
        query = self._build_query(event)
        retrieved_documents = await self.__search_documents(
            query, k, event, self.vectorstore
        )

        return retrieved_documents
