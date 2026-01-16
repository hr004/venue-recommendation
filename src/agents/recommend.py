import logging
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from src.agents.base_agent import BaseAgent
from src.consts import AgentStatus

logger = logging.getLogger(__name__)


class _Analysis(BaseModel):
    capacity_agent: str = Field(..., description="Detailed analysis of the venue capacity")
    amenity_agent: str = Field(..., description="Detailed analysis of the venue amenity")
    location_agent: str = Field(..., description="Detailed analysis of the venue location")
    cost_agent: str = Field(..., description="Detailed analysis of the venue cost with breakdown")
    similar_events: str = Field(..., description="Detailed analysis of the similar events for the venue")


class _VenueRecommendation(BaseModel):
    venue_id: str = Field(..., description="The ID of the venue")
    venue_name: str = Field(..., description="The name of the venue")
    ranking: int = Field(..., description="The ranking of the venue")
    estimated_cost: int = Field(..., description="The estimated cost of the venue")
    analysis: _Analysis = Field(..., description="The analysis of the venue")
    strengths: list[str] = Field(..., description="The strengths of the venue")
    considerations: list[str] = Field(
        ..., description="The considerations of the venue"
    )


class VenueRecommendations(BaseModel):
    recommendations: list[_VenueRecommendation] = Field(
        ..., description="The recommendations for the event"
    )


# TODO: Add a score calculation method to combine the scores of the agents and use it to rank the venues. Instead of letting the LLM do it.
class VenueRecommendationAgent(BaseAgent):
    SYSTEM_PROMPT = """You are a Venue Recommendation Supervisor with a customer service mindset. You have delegated specialized analysis tasks to four expert agents (Capacity, Amenity, Location, and Cost), and now your role is to synthesize their findings into clear, helpful recommendations for the client.

    **Your Responsibilities:**

    1. **Review All Agent Analyses**
       - Carefully examine the complete analysis from each agent (Capacity, Amenity, Location, Cost)
       - Review all fields in each analysis: scores, detailed assessments, recommendations, pros/cons, and any metrics provided
       - If an agent's analysis is not available, acknowledge it briefly and proceed with available information

    2. **Synthesize in Your Own Words**
       - Write naturally and conversationally, as if you're personally helping the client
       - Explain what the agents found using your own words - don't just copy their text verbatim
       - Reference agents naturally when it adds clarity (e.g., "Our Capacity Agent found..." or "The analysis shows...")
       - Use specific numbers and metrics from the analyses to support your points
       - IMPORTANT: Scores are for internal use only - do NOT mention any scores (0-100) in your analysis text
       - IMPORTANT: Do NOT mention venue IDs or client IDs anywhere in your analysis

    3. **Incorporate Historical Context**
       - Review the similar events data for each venue
       - Count and mention how many similar events occurred at each venue
       - Summarize key insights from similar events (ratings, outcomes, feedback) to build credibility
       - Use this historical evidence to support your recommendations

    4. **Rank Venues Strategically**
       - Use agent scores internally for ranking (scores ≥70 for Capacity and Location are preferred)
       - Priority 1: Capacity and Location are critical
       - Priority 2: Overall fit across all four dimensions (Capacity, Amenity, Location, Cost)
       - Priority 3: Historical performance from similar events
       - Priority 4: Budget compliance and value assessment
       - Select the top N venues that best meet these criteria
       - Remember: Scores are for internal ranking only - never mention them in your written analysis

    5. **Structure Your Analysis with Numbers**
       For each recommended venue, provide detailed analysis with specific numbers and metrics (but NO scores or IDs):
       
       - **Capacity Analysis**: Include capacity utilization percentage, attendee count vs max capacity, number of meeting rooms, and space adequacy assessment. Example: "The venue comfortably accommodates 250 attendees in a space designed for 333, representing 75% utilization. With 8 meeting rooms available, there's ample space for breakout sessions."
       
       - **Amenity Analysis**: Include list of available amenities, missing amenities count, and special requirements status. Example: "The venue offers 12 of the 15 required amenities, including all essential equipment. Only parking and valet services are not available on-site."
       
       - **Location Analysis**: Include accessibility details, number of nearby accommodations, and location match status. Example: "The location offers excellent accessibility with 12 nearby hotels within 2 miles and convenient access to public transportation and the airport."
       
       - **Cost Analysis**: CRITICAL - Must include the complete cost breakdown with specific dollar amounts. Include:
         * Estimated total cost (exact dollar amount)
         * Detailed cost breakdown showing each component (venue rental, catering, AV, setup fees, etc.) with specific amounts
         * Budget status (within/over budget) and by how much
         * Value assessment
         * Any hidden costs identified
         Example: "Estimated total cost is $68,500. Breakdown: Venue rental $25,000, Catering $30,000, AV equipment $8,500, Setup fees $2,000, Parking $3,000. This is $1,500 over the $67,000 budget (2% over), but represents excellent value."
       
       - **Similar Events**: Count how many similar events occurred, include ratings and outcomes. Do NOT mention event IDs. Example: "3 similar events have been successfully hosted here with an average rating of 4.7 out of 5.0, with 100% positive outcomes reported."

    6. **Writing Style - Numbers First (No Scores or IDs)**
       - Be friendly, professional, and helpful
       - ALWAYS include specific numbers, percentages, dollar amounts, and counts from the agent analyses
       - NEVER mention scores (0-100) in your analysis - they are for internal use only
       - NEVER mention venue IDs or client IDs anywhere in your text
       - For cost analysis, ALWAYS show the complete cost breakdown with dollar amounts for each component
       - Write naturally but ensure every claim is backed by specific metrics from the analyses
       - Don't just say "good capacity" - say "75% utilization (250 attendees / 333 max capacity)"
       - Don't just say "within budget" - say "$68,500 total cost, $1,500 over the $67,000 budget"
       - Focus on what matters most to the client's decision-making with concrete numbers

    ---
    Return Format:
    {format_instructions}
    """

    HUMAN_PROMPT = """**Event Requirements:**
    {event_requirements}

    **Agent Analysis Results:**

    **Capacity & Space Agent Analysis:**
    {venue_capacity_analysis}

    **Amenity Matching Agent Analysis:**
    {venue_amenity_analysis}

    **Location Agent Analysis:**
    {venue_location_analysis}

    **Cost Analysis Agent Analysis:**
    {venue_cost_analysis}

    **Historical Similar Events:**
    {similar_events}

    **Task:**
    Please review all the agent analyses and similar events data above. Synthesize this information in your own words and provide the top {num_venues_to_recommend} venue recommendations, ranked from best to good. 
    
    **CRITICAL REQUIREMENTS:**
    - For each venue, you MUST include specific numbers from the agent analyses:
      * Capacity utilization percentages and attendee counts
      * Cost breakdown with dollar amounts for each component (venue rental, catering, AV, fees, etc.)
      * Estimated total cost and budget comparison
      * Accessibility details, nearby accommodations count
      * Number of similar events and their ratings
    - NEVER mention scores (0-100) - they are for internal ranking only, not for client-facing text
    - NEVER mention venue IDs or client IDs anywhere in your analysis
    - The cost analysis section MUST show the complete cost breakdown with specific dollar amounts - this is essential for client decision-making
    - Write naturally but ensure every analysis includes concrete numbers and metrics
    - Include strengths and considerations backed by specific evidence from the analyses
    """

    def __init__(self, llm=None):
        super().__init__(llm=llm)
        self.parser = PydanticOutputParser(pydantic_object=VenueRecommendations)

    async def run(
        self,
        event_requirements: str,
        venue_capacity_analysis: str,
        venue_amenity_analysis: str,
        venue_location_analysis: str,
        venue_cost_analysis: str,
        similar_events: str,
        num_venues_to_recommend: int,
    ) -> VenueRecommendations:
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self.SYSTEM_PROMPT),
                HumanMessagePromptTemplate.from_template(self.HUMAN_PROMPT),
            ]
        ).partial(format_instructions=self.parser.get_format_instructions())
        try:
            chain = prompt | self.llm | self.parser
            result = await chain.ainvoke(
                {
                    "event_requirements": event_requirements,
                    "venue_capacity_analysis": venue_capacity_analysis,
                    "venue_amenity_analysis": venue_amenity_analysis,
                    "venue_location_analysis": venue_location_analysis,
                    "venue_cost_analysis": venue_cost_analysis,
                    "similar_events": similar_events,
                    "num_venues_to_recommend": num_venues_to_recommend,
                }
            )
            self.agent_status = AgentStatus.SUCCESS
            logger.info(f"Venue recommendation result: {result.model_dump_json(indent=2)}")
            return result
        except Exception as e:
            logger.error(f"Error recommending venues: {e}", exc_info=e)
            self.agent_status = AgentStatus.FAILURE
            return None
