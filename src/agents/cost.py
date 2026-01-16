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
        description="Short summary of cost related advantages of the venue for the event",
        default="",
    )
    cons: str = Field(
        description="Short summary of cost related disadvantages of the venue for the event",
        default="",
    )


class _CostAnalysis(BaseModel):
    """Analysis result from the Cost Analysis Agent"""

    score: int = Field(
        description="Score of the venue matching the event budget", ge=0, le=100
    )
    analysis: str = Field(
        description="Short summary analysis of cost matching between event budget and venue pricing",
    )
    budget_met: bool = Field(
        description="Whether the estimated cost is within the event budget"
    )
    estimated_total_cost: float = Field(
        description="Estimated total cost for the event at this venue"
    )
    cost_breakdown: Dict[str, float] = Field(
        description="Breakdown of available costs from the available information",
        default_factory=dict,
    )
    value_assessment: str = Field(
        description="Assessment of value (excellent, good, fair, poor)"
    )
    hidden_costs: List[str] = Field(
        description="List of potential hidden costs to consider", default_factory=list
    )
    recommendation: _Recommendation = Field(
        description="Recommendation for the venue in terms of cost matching"
    )


class VenueCostAnalysis(BaseModel):
    cost_analysis: List[_CostAnalysis] = Field(
        description="List of cost analyses for each venue"
    )


class CostAnalysisAgent(BaseAgent):
    """
    Cost Analysis Agent - Analyzes budget fit, value assessment, and hidden costs.
    """

    SYSTEM_PROMPT = """You are an expert Cost Analysis Agent for event venue recommendations.

        Your task is to analyze how well each venue's pricing fits an event's budget, considering:
        1. Event budget and budget flexibility
        2. Venue daily rates and fees
        3. Estimated costs for catering, AV, and other services
        4. Value assessment compared to similar venues
        5. Hidden costs (parking, setup fees, cancellation policies, etc.)
        6. Historical cost patterns from similar events
        7. Score should be higher if it matches client budget and venue pricing.

        Use the retrieved similar event history to understand:
        - What cost ranges worked well for similar events at each venue
        - Common cost-related challenges
        - Success factors related to budget management
        - Feedback about pricing and value

        Provide a comprehensive analysis that:
        - Estimates total cost based on event duration and requirements
        - Evaluates if the cost fits within the budget
        - Assesses value compared to similar venues
        - Identifies potential hidden costs
        - References successful similar events when relevant
        - Provides actionable recommendations

        Be specific and to the point. Reference the retrieved historical events when making your analysis.

        ---
        Return Format Instructions:
        Do not include commas in the numbers.
        {format_instructions}
    """

    HUMAN_PROMPT = """Event Budget Requirements:
        {event_budget}

        Venue Pricing:
        {venue_pricing}

        Retrieved Similar Events (for context):
        {similar_events}

        Analyze the cost and budget matching and provide your assessment in the required format.
    """

    def __init__(self, llm=None):
        super().__init__(llm=llm)
        self.parser = PydanticOutputParser(pydantic_object=VenueCostAnalysis)

    def _format_event_budget(self, event: Dict[str, Any]) -> str:
        """Format event budget requirements for the prompt"""
        budget = event.get("budget", 0)
        budget_flexibility = event.get("budget_flexibility", "firm")
        duration_days = event.get("duration_days", 1)

        budget_info = [
            f"Event Budget: ${budget:,}",
            f"Budget Flexibility: {budget_flexibility}",
            f"Duration: {duration_days} days",
        ]

        return "\n".join(budget_info)

    def _format_venue_pricing(self, venue: Dict[str, Any]) -> str:
        """Format venue pricing information for the prompt"""
        daily_rate = venue.get("daily_rate", 0)
        half_day_rate = venue.get("half_day_rate", 0)
        setup_fee = venue.get("setup_fee", 0)
        av_included = venue.get("av_included", False)
        cancellation_policy = venue.get("cancellation_policy", "")

        venue_pricing_info = [
            f"Venue: {venue.get('name', 'Unknown')}",
            f"Daily Rate: ${daily_rate:,}",
            f"Half-Day Rate: ${half_day_rate:,}",
            f"Setup Fee: ${setup_fee:,}",
            f"AV Equipment Included: {av_included}",
            f"Cancellation Policy: {cancellation_policy}",
        ]
        return "\n".join(venue_pricing_info)

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

    async def _analyze_cost_matching(
        self,
        event: Dict[str, Any],
        venues: List[Dict[str, Any]],
        retrieved_documents: List[Document],
    ) -> VenueCostAnalysis:
        """Perform cost and budget matching analysis using LLM"""

        # Format inputs
        event_budget = self._format_event_budget(event)
        venue_pricing_info = [self._format_venue_pricing(venue) for venue in venues]
        similar_events = self._format_similar_events(retrieved_documents)

        # Create prompt with format instructions
        system_message = SystemMessagePromptTemplate.from_template(self.SYSTEM_PROMPT)
        human_message = HumanMessagePromptTemplate.from_template(self.HUMAN_PROMPT)

        prompt = ChatPromptTemplate.from_messages(
            [system_message, human_message]
        ).partial(format_instructions=self.parser.get_format_instructions())

        # Create chain with parser
        chain = prompt | self.llm | self.parser
        try:
            result = await chain.ainvoke(
                {
                    "event_budget": event_budget,
                    "venue_pricing": venue_pricing_info,
                    "similar_events": similar_events,
                }
            )
            self.agent_status = AgentStatus.SUCCESS
            return result
        except Exception as e:
            logger.error(f"Error analyzing cost matching: {e}", exc_info=e)
            self.agent_status = AgentStatus.FAILURE
            return None

    async def run(
        self,
        event: Dict[str, Any],
        retrieved_documents: List[Document] = None,
    ) -> VenueCostAnalysis:
        """
        Run full cost analysis and return structured result.

        Args:
            event: Event request data
            venue: Venue data
            retrieved_documents: Similar events from RAG search

        Returns:
            CostAnalysis object with detailed results
        """
        if retrieved_documents is None:
            retrieved_documents = []

        # each of the document has venues and clients
        venues = [
            doc.metadata.get("venue")
            for doc in retrieved_documents
            if doc.metadata.get("venue") is not None
        ]
        result = await self._analyze_cost_matching(event, venues, retrieved_documents)
        logger.info(f"Cost analysis result: {result.model_dump_json(indent=2)}")
        return result
