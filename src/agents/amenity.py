import logging
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from pydantic import BaseModel, Field
from src.agents.base_agent import BaseAgent
from src.consts import AgentStatus

logger = logging.getLogger(__name__)


class _Recommendation(BaseModel):
    recommend: bool = Field(description="Whether to recommend the venue for the event")
    pros: str = Field(description="Pros of the venue for the event")
    cons: str = Field(description="Cons of the venue for the event")


class _VenueAmenityAnalysis(BaseModel):
    """Analysis result from the Amenity Matching Agent"""

    score: int = Field(
        description="Score of the venue matching the event requirements", ge=0, le=100
    )
    analysis: str = Field(
        description="Short summary analysis of amenity matching between event requirements and venue amenities"
    )
    venue_name: str = Field(description="Name of the venue")
    venue_id: str = Field(description="Venue ID")
    required_amenities_match: bool = Field(
        description="Whether all required amenities are available"
    )
    missing_amenities: List[str] = Field(
        description="List of required amenities that are missing", default_factory=list
    )
    available_amenities: List[str] = Field(
        description="List of amenities that are available and match requirements",
        default_factory=list,
    )
    special_requirements_status: str = Field(
        description="Status of special requirements (met, partially_met, not_met)"
    )
    recommendation: _Recommendation = Field(description="Recommendation for the venue")


class VenueAmenityAnalysis(BaseModel):
    venue_amenity_analysis: List[_VenueAmenityAnalysis] = Field(
        description="List of venue amenity analyses"
    )


class AmenityMatchingAgent(BaseAgent):
    """
    Amenity Matching Agent - Analyzes required equipment, services, and catering.
    """

    SYSTEM_PROMPT = """You are an expert Amenity Matching Agent for event venue recommendations.

        Your task is to analyze how well a venue's amenities match an event's requirements, considering:
        1. Required amenities (must-have equipment and services)
        2. Preferred amenities (nice-to-have)
        3. Special requirements (dietary restrictions, accessibility, etc.)
        4. Historical success patterns from similar events
        5. Score should be higher if it matches client preferences and special requirements.

        Use the retrieved similar event history to understand:
        - What amenities worked well for similar events
        - Common challenges with amenity mismatches
        - Success factors related to amenities
        - Feedback about amenity quality

        Provide a comprehensive analysis that:
        - Confirms which required amenities are available
        - Identifies any missing critical amenities
        - Evaluates special requirements (catering needs, accessibility, etc.)
        - References successful similar events when relevant

        Be specific and reference the retrieved historical events when making your analysis.

        ---
        Return Format:
        {format_instructions}
    """

    HUMAN_PROMPT = """Event Requirements:
        {event_requirements}

        Venue Amenities:
        {venue_amenities}

        Retrieved Similar Events (for context):
        {similar_events}

        Analyze the amenity matching and provide your assessment in the required format.
    """

    def __init__(self, llm=None):
        super().__init__(llm=llm)
        self.parser = PydanticOutputParser(pydantic_object=VenueAmenityAnalysis)

    def _format_event_requirements(self, event: Dict[str, Any]) -> str:
        """Format event requirements for the prompt"""
        required = event.get("required_amenities", [])
        preferred = event.get("preferred_amenities", [])
        special = event.get("special_requirements", [])
        event_style = event.get("event_style", "")

        parts = [
            f"Event Type: {event.get('event_type', 'unknown')}",
            f"Event Style: {event_style}",
            f"Required Amenities: {', '.join(required) if required else 'None specified'}",
        ]

        if preferred:
            parts.append(f"Preferred Amenities: {', '.join(preferred)}")

        if special:
            parts.append(f"Special Requirements: {', '.join(special)}")

        return "\n".join(parts)

    def _format_venue_amenities(self, venue: Dict[str, Any]) -> str:
        """Format venue amenities for the prompt"""
        amenities = venue.get("amenities", [])
        features = venue.get("features", [])
        catering_options = venue.get("catering_options", [])
        av_included = venue.get("av_included", False)

        parts = [
            f"Venue: {venue.get('name', 'Unknown')}",
            f"Available Amenities: {', '.join(amenities) if amenities else 'None listed'}",
        ]

        if features:
            parts.append(f"Features: {', '.join(features)}")

        if catering_options:
            parts.append(f"Catering Options: {', '.join(catering_options)}")

        if av_included:
            parts.append("AV Equipment: Included in-house")
        else:
            parts.append("AV Equipment: May require external rental")

        return "\n".join(parts)

    def _format_similar_events(self, retrieved_documents: List[Document]) -> str:
        """Format retrieved similar events for context"""
        if not retrieved_documents:
            return "No similar events found in history."

        formatted_events = []
        for i, doc in enumerate(retrieved_documents, 1):  # Limit to top 5

            event_info = f"""
            Similar Event {i}:
            {doc.page_content}
            """
            formatted_events.append(event_info)

        return "\n".join(formatted_events)

    async def _analyze_amenity_matching(
        self,
        event: Dict[str, Any],
        venues: List[Dict[str, Any]],
        retrieved_documents: List[Document],
    ) -> VenueAmenityAnalysis:
        """Perform venue amenity analysis"""
        event_reqs = self._format_event_requirements(event)
        venue_amenities = [self._format_venue_amenities(venue) for venue in venues]
        similar_events = self._format_similar_events(retrieved_documents)

        system_message = SystemMessagePromptTemplate.from_template(self.SYSTEM_PROMPT)
        human_message = HumanMessagePromptTemplate.from_template(self.HUMAN_PROMPT)

        prompt = ChatPromptTemplate.from_messages(
            [system_message, human_message]
        ).partial(format_instructions=self.parser.get_format_instructions())

        chain = prompt | self.llm | self.parser

        try:
            result = await chain.ainvoke(
                {
                    "event_requirements": event_reqs,
                    "venue_amenities": venue_amenities,
                    "similar_events": similar_events,
                    "num_venues": len(venues),
                }
            )
            self.agent_status = AgentStatus.SUCCESS
            return result
        except Exception as e:
            logger.error(f"Error analyzing amenity matching: {e}", exc_info=e)
            raise e

    async def run(
        self,
        event: Dict[str, Any],
        retrieved_documents: List[Document] = None,
    ) -> VenueAmenityAnalysis:
        """
        Run full amenity analysis and return structured result.

        Args:
            event: Event request data
            retrieved_documents: Similar events from RAG search

        Returns:
            VenueAmenityAnalysis object with detailed results
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
        result = await self._analyze_amenity_matching(
            event, venues, retrieved_documents
        )
        logger.info(f"Amenity analysis result: {result.model_dump_json(indent=2)}")
        return result
