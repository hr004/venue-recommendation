import logging
from typing import Any, Dict, List
from src.agents.base_agent import BaseAgent
from src.consts import AgentStatus
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

logger = logging.getLogger(__name__)


class _Recommendation(BaseModel):
    recommend: bool = Field(description="Whether to recommend the venue for the event")
    pros: str = Field(
        description="Short summary of location related advantages of the venue for the event"
    )
    cons: str = Field(
        description="Short summary of location related disadvantages of the venue for the event"
    )


class _LocationAnalysis(BaseModel):
    """Analysis result from the Location Agent"""

    score: int = Field(
        description="Score of the venue matching the event location", ge=0, le=100
    )
    analysis: str = Field(
        description="Short summary analysis of location matching between event preferences and venue location"
    )
    venue_name: str = Field(description="Name of the venue")
    venue_id: str = Field(description="Venue ID")
    location_match: bool = Field(
        description="Whether the venue location matches the event's location preferences"
    )
    region_match: bool = Field(
        description="Whether the venue is in the preferred region"
    )
    accessibility_score: float = Field(
        description="Accessibility score (0-1) based on airport distance, public transit, etc."
    )
    nearby_accommodations: int = Field(
        description="Number of nearby hotels/accommodations"
    )
    recommendations: _Recommendation = Field(description="Recommendation for the venue")


class VenueLocationAnalysis(BaseModel):
    location_analysis: List[_LocationAnalysis] = Field(
        description="List of location analyses for each venue"
    )


class LocationAgent(BaseAgent):
    """
    Location Agent - Analyzes geographic preferences, accessibility, and nearby accommodations.
    """

    SYSTEM_PROMPT = """You are an expert Location Agent for event venue recommendations.

        Your task is to analyze how well a venue's location matches an event's geographic preferences, considering:
        1. Location preference (region, cities)
        2. Airport distance and accessibility
        3. Public transit availability
        4. Nearby hotels and accommodations
        5. Historical success patterns from similar events in the same location
        6. There are {num_venues} venues in total. You should analyze the location for each venue separately.
        7. Then pick top 10 venues that are most suitable for the event and provide the analysis for each of them.
        8. Score should be higher if it matches client location preferences and venue location.

        Use the retrieved similar event history to understand:
        - What locations worked well for similar events
        - Common location-related challenges
        - Success factors related to location and accessibility
        - Feedback about location convenience

        Provide a comprehensive analysis that:
        - Evaluates if the venue location matches the event's preferences
        - Assesses accessibility (airport distance, public transit)
        - Considers nearby accommodations for attendees
        - References successful similar events in the same location when relevant
        - Provides actionable recommendations

        Be specific and reference the retrieved historical events when making your analysis.

        ---
        Return Format:
        {format_instructions}
    """

    HUMAN_PROMPT = """Event Location Requirements:
        {event_location}

        Venue Location:
        {venue_location}

        Retrieved Similar Events (for context):
        {similar_events}

        Analyze the location matching and provide your assessment in the required format.
    """

    def __init__(self, llm=None):
        super().__init__(llm=llm)
        self.parser = PydanticOutputParser(pydantic_object=VenueLocationAnalysis)

    def _format_event_location(self, event: Dict[str, Any]) -> str:
        """Format event location requirements for the prompt"""
        location_pref = event.get("location_preference", "")
        location_reqs = event.get("location_requirements", {})
        regions = location_reqs.get("region", [])
        cities = location_reqs.get("cities", [])
        max_airport_distance = location_reqs.get("max_airport_distance_miles", None)

        parts = [
            f"Location Preference: {location_pref}",
        ]

        if regions:
            parts.append(f"Preferred Regions: {', '.join(regions)}")

        if cities:
            parts.append(f"Preferred Cities: {', '.join(cities)}")

        if max_airport_distance:
            parts.append(f"Maximum Airport Distance: {max_airport_distance} miles")

        return "\n".join(parts)

    def _format_venue_location(self, venue: Dict[str, Any]) -> str:
        """Format venue location information for the prompt"""
        city = venue.get("city", "")
        state = venue.get("state", "")
        region = venue.get("region", "")
        address = venue.get("address", "")
        airport_distance = venue.get("airport_distance_miles", None)
        public_transit = venue.get("public_transit", False)
        nearby_hotels = venue.get("nearby_hotels", 0)

        parts = [
            f"Venue: {venue.get('name', 'Unknown')}",
            f"City: {city}",
        ]

        if state:
            parts.append(f"State: {state}")

        if region:
            parts.append(f"Region: {region}")

        if address:
            parts.append(f"Address: {address}")

        if airport_distance is not None:
            parts.append(f"Airport Distance: {airport_distance} miles")

        parts.append(f"Public Transit Available: {public_transit}")

        if nearby_hotels:
            parts.append(f"Nearby Hotels: {nearby_hotels}")

        return "\n".join(parts)

    def _format_similar_events(self, retrieved_documents: List[Document]) -> str:
        """Format retrieved similar events for context"""
        if not retrieved_documents:
            return "No similar events found in history."

        formatted_events = []
        for i, doc in enumerate(retrieved_documents, 1):
            event_info = f"""
            Similar Event {i}:
            {doc.page_content}
            """
            formatted_events.append(event_info)

        return "\n".join(formatted_events)

    async def _analyze_location_matching(
        self,
        event: Dict[str, Any],
        venues: List[Dict[str, Any]],
        retrieved_documents: List[Document],
    ) -> VenueLocationAnalysis:
        """Perform location matching analysis using LLM"""

        event_location = self._format_event_location(event)
        venue_location = [self._format_venue_location(venue) for venue in venues]
        similar_events = self._format_similar_events(retrieved_documents)

        system_message = SystemMessagePromptTemplate.from_template(self.SYSTEM_PROMPT)
        human_message = HumanMessagePromptTemplate.from_template(self.HUMAN_PROMPT)

        prompt = ChatPromptTemplate.from_messages(
            [system_message, human_message]
        ).partial(format_instructions=self.parser.get_format_instructions())

        chain = prompt | self.llm | self.parser
        result = await chain.ainvoke(
            {
                "event_location": event_location,
                "venue_location": venue_location,
                "similar_events": similar_events,
                "num_venues": len(venue_location),
            }
        )

        try:
            result = await chain.ainvoke(
                {
                    "event_location": event_location,
                    "venue_location": venue_location,
                    "similar_events": similar_events,
                    "num_venues": len(venue_location),
                }
            )
            self.agent_status = AgentStatus.SUCCESS
            return result
        except Exception as e:
            logger.error(f"Error analyzing location matching: {e}", exc_info=e)
            self.agent_status = AgentStatus.FAILURE
            return None

    async def run(
        self,
        event: Dict[str, Any],
        retrieved_documents: List[Document] = None,
    ) -> VenueLocationAnalysis:
        """
        Run full location analysis and return structured result.

        Args:
            event: Event request data
            venue: Venue data
            retrieved_documents: Similar events from RAG search

        Returns:
            VenueLocationAnalysis object with detailed results
        """
        if retrieved_documents is None:
            retrieved_documents = []

        # each of the document has venues and clients
        venues = {
            venue["venue_id"]: venue
            for doc in retrieved_documents
            if (venue := doc.metadata.get("venue")) is not None
            and venue.get("venue_id")
        }.values()
        venues = list(venues)
        result = await self._analyze_location_matching(
            event, venues, retrieved_documents
        )
        logger.info(f"Location analysis result: {result.model_dump_json(indent=2)}")
        return result
