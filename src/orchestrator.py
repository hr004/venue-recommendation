import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.documents import Document
from src.consts import AgentStatus
from src.agents import (
    AmenityMatchingAgent, 
    CapacitySpaceAgent,
    CostAnalysisAgent,
    LocationAgent,
    VenueRecommendationAgent
)

logger = logging.getLogger(__name__)


class VenueRecommendationOrchestrator:
    """
    Orchestrator that coordinates all specialized agents and synthesizes their findings.
    Includes error handling and recovery - continues processing even if individual agents fail.
    """

    def __init__(self):
        self.capacity_agent = CapacitySpaceAgent()
        self.amenity_agent = AmenityMatchingAgent()
        self.location_agent = LocationAgent()
        self.cost_agent = CostAnalysisAgent()
        self.supervisor_agent = VenueRecommendationAgent()

    async def recommend(
        self, event, retrieved_documents: List[Document], top_n: int = 3
    ) -> List[str]:
        """
        Recommend venues based on all agent analyses.
        Continues processing even if individual agents fail.

        Args:
            retrieved_documents: Similar events from RAG search

        Returns:
            List of venue recommendations
        """

        results = await asyncio.gather(
            self.amenity_agent.run(event, retrieved_documents),
            self.cost_agent.run(event, retrieved_documents),
            self.location_agent.run(event, retrieved_documents),
            self.capacity_agent.run(event, retrieved_documents),
        )
        recommendations = {}
        venue_amenity_analysis, cost_analysis, location_analysis, capacity_analysis = (
            results
        )

        agent_results = [
            ("amenity", self.amenity_agent, venue_amenity_analysis),
            ("cost", self.cost_agent, cost_analysis),
            ("location", self.location_agent, location_analysis),
            ("capacity", self.capacity_agent, capacity_analysis),
        ]

        for agent_name, agent, result in agent_results:
            if result is not None and agent.agent_status == AgentStatus.SUCCESS:
                recommendations[agent_name] = result

        if not any([result for _, _, result in agent_results]):
            retry_tasks = [
                (agent_name, agent, result)
                for agent_name, agent, result in agent_results
                if result is None or agent.agent_status == AgentStatus.FAILURE
            ]

            if retry_tasks:
                retry_results = await asyncio.gather(
                    *[
                        self._retry_agent(agent, agent_name, event, retrieved_documents)
                        for agent_name, agent, result in retry_tasks
                    ]
                )
                for agent_name, agent, result in retry_tasks:
                    if result is not None and agent.agent_status == AgentStatus.SUCCESS:
                        recommendations[agent_name] = result
                return recommendations

        similar_events = "\n".join([doc.page_content for doc in retrieved_documents])
        supervisor_result = await self.supervisor_agent.run(
            event,
            recommendations.get("capacity", ""),
            recommendations.get("amenity", ""),
            recommendations.get("location", ""),
            recommendations.get("cost", ""),
            similar_events,
            top_n,
        )
        return supervisor_result

    async def _retry_agent(
        self,
        agent,
        agent_name: str,
        event: Dict[str, Any],
        retrieved_documents: List[Document],
        max_retries: int = 3,
    ) -> Optional[Any]:
        """
        Retry a failed agent up to max_retries times.

        Args:
            agent: The agent instance to retry
            agent_name: Name of the agent (for logging)
            event: Event request data
            retrieved_documents: Similar events from RAG search
            max_retries: Maximum number of retry attempts

        Returns:
            Agent result if successful, None if all retries failed
        """
        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Retrying {agent_name} (attempt {attempt + 1}/{max_retries})"
                )
                result = await agent.run(event, retrieved_documents)

                if result is not None and agent.agent_status == AgentStatus.SUCCESS:
                    logger.info(
                        f"{agent_name} succeeded on retry attempt {attempt + 1}"
                    )
                    return result
            except Exception as e:
                logger.warning(f"{agent_name} retry {attempt + 1} failed: {e}")

        logger.error(f"{agent_name} failed after {max_retries} retry attempts")
        return None
