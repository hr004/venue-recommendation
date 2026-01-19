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
        description="Short summary of capacity related advantages of the venue for the event"
    )
    cons: str = Field(
        description="Short summary of capacity related disadvantages of the venue for the event"
    )


class _CapacityAnalysis(BaseModel):
    """Analysis result from the Capacity & Space Agent"""

    score: int = Field(
        description="Score of the venue matching the event capacity", ge=0, le=100
    )
    analysis: str = Field(
        description="Short summary analysis of capacity and space matching between event requirements and venue capabilities"
    )
    venue_name: str = Field(description="Name of the venue")
    venue_id: str = Field(description="Venue ID")
    capacity_suitable: bool = Field(
        description="Whether the venue capacity is suitable for the event"
    )
    capacity_utilization: float = Field(
        description="Percentage of venue capacity that will be used (attendee_count / max_capacity)"
    )
    meeting_rooms_sufficient: bool = Field(
        description="Whether the number of meeting rooms is sufficient for breakout sessions"
    )
    space_adequacy: str = Field(
        description="Assessment of space adequacy (excellent, good, adequate, tight, insufficient)"
    )
    recommendations: _Recommendation = Field(
        description="Recommendation based on capacity and space analysis",
    )


class VenueCapacityAnalysis(BaseModel):
    capacity_analysis: List[_CapacityAnalysis] = Field(
        description="List of capacity analyses for each given venue"
    )


class CapacitySpaceAgent(BaseAgent):
    """
    Capacity & Space Agent - Analyzes attendee fit, room layouts, and breakout spaces.
    """

    SYSTEM_PROMPT = """You are an expert Capacity & Space Agent for event venue recommendations.

        Your task is to analyze how well a venue's capacity and space configuration matches an event's requirements, considering:
        1. Attendee count vs venue capacity (optimal utilization is typically 70-90%)
        2. Number of meeting rooms for breakout sessions
        3. Room layouts and flexibility
        4. Space for networking, registration, and other activities
        5. Historical success patterns from similar events
        6. Score should be higher if it matches client capacity and space requirements.

        Use the retrieved similar event history to understand:
        - What capacity ranges worked well for similar events
        - Common space-related challenges
        - Success factors related to space and layout
        - Feedback about capacity utilization

        Provide a comprehensive analysis that:
        - Evaluates if the venue capacity is appropriate (not too small, not too large)
        - Assesses meeting room availability for breakout sessions
        - Considers space for registration, networking, and other activities
        - References successful similar events when relevant
        - Provides actionable recommendations

        Be specific and reference the retrieved historical events when making your analysis.

        ---
        Return Format:
        {format_instructions}
    """

    HUMAN_PROMPT = """Event Requirements:
        {event_requirements}

        Capacity & Space for each venue:
        {venue_capacity}
        There are {num_venues} venues in total. 
        You should analyze the capacity and space for each venue separately.

        Retrieved Similar Events (for context):
        {similar_events}

        Analyze the capacity and space matching and provide your assessment in the required format.
    """

    def __init__(self, llm=None):
        super().__init__(llm)
        self.parser = PydanticOutputParser(pydantic_object=VenueCapacityAnalysis)

    def _format_event_requirements(self, event: Dict[str, Any]) -> str:
        """Format event requirements for the prompt"""
        attendee_count = event.get("attendee_count", 0)
        duration_days = event.get("duration_days", 0)
        event_type = event.get("event_type", "unknown")

        parts = [
            f"Event Type: {event_type}",
            f"Attendee Count: {attendee_count}",
            f"Duration: {duration_days} days",
        ]

        special_reqs = event.get("special_requirements", [])
        if special_reqs:
            parts.append(f"Special Requirements: {', '.join(special_reqs)}")
        breakout_requirements = [
            req for req in special_reqs if "breakout" in req.lower()
        ]
        if breakout_requirements:
            parts.append(
                f"Breakout Room Requirements: {', '.join(breakout_requirements)}"
            )

        return "\n".join(parts)

    def _format_venue_capacity(self, venue: Dict[str, Any]) -> str:
        """Format venue capacity information for the prompt"""
        max_capacity = venue.get("max_capacity", 0)
        min_capacity = venue.get("min_capacity", 0)
        meeting_rooms = venue.get("meeting_rooms", 0)
        total_sqft = venue.get("total_sqft", 0)
        largest_room_sqft = venue.get("largest_room_sqft", 0)
        ballroom_capacity = venue.get("ballroom_capacity", 0)

        venue_capacity_info = [
            f"Venue: {venue.get('name', 'Unknown')}",
            f"Maximum Capacity: {max_capacity}",
            f"Minimum Capacity: {min_capacity}",
            f"Meeting Rooms: {meeting_rooms}",
        ]

        if total_sqft:
            venue_capacity_info.append(f"Total Square Footage: {total_sqft:,} sqft")

        if largest_room_sqft:
            venue_capacity_info.append(f"Largest Room: {largest_room_sqft:,} sqft")

        if ballroom_capacity:
            venue_capacity_info.append(f"Ballroom Capacity: {ballroom_capacity}")

        return "\n".join(venue_capacity_info)

    def _format_similar_events(self, retrieved_documents: List[Document]) -> str:
        """Format retrieved similar events for context"""
        if not retrieved_documents:
            return "No similar events found in history."

        formatted_events = []
        for i, doc in enumerate(retrieved_documents[:5], 1):
            event_info = f"""
            Similar Event {i}:
            {doc.page_content}
            """
            formatted_events.append(event_info)

        return "\n".join(formatted_events)

    async def _analyze_capacity_matching(
        self,
        event: Dict[str, Any],
        venues: List[Dict[str, Any]],
        retrieved_documents: List[Document],
    ) -> VenueCapacityAnalysis:
        """Perform capacity and space matching analysis using LLM"""

        event_reqs = self._format_event_requirements(event)
        venue_capacity = [self._format_venue_capacity(venue) for venue in venues]
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
                    "venue_capacity": venue_capacity,
                    "similar_events": similar_events,
                    "num_venues": len(venues),
                }
            )
            self.agent_status = AgentStatus.SUCCESS
            return result
        except Exception as e:
            logger.error(f"Error analyzing capacity matching: {e}", exc_info=e)
            self.agent_status = AgentStatus.FAILURE
            return None

    async def run(
        self,
        event: Dict[str, Any],
        retrieved_documents: List[Document] = None,
    ) -> VenueCapacityAnalysis:
        """
        Run full capacity and space analysis and return structured result.

        Args:
            event: Event request data
            venue: Venue data
            retrieved_documents: Similar events from RAG search

        Returns:
            VenueCapacityAnalysis object with detailed results
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
        result = await self._analyze_capacity_matching(
            event, venues, retrieved_documents
        )
        logger.info(f"Capacity analysis result: {result.model_dump_json(indent=2)}")
        return result

