from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.config import config
from src.consts import AgentStatus


class BaseAgent:
    def __init__(self, llm=None):
        if llm is None:
            self.llm = ChatOpenAI(
                model=config.llm_config.model,
                temperature=0,
                max_tokens=6000,
            )
        else:
            self.llm = llm

        self._agent_status = AgentStatus.PENDING

    @abstractmethod
    async def run(
        self, event: Dict[str, Any], retrieved_documents: List[Document]
    ) -> Any:
        raise NotImplementedError("Subclasses must implement run method")

    @property
    def agent_status(self) -> AgentStatus:
        return self._agent_status

    @agent_status.setter
    def agent_status(self, status: AgentStatus):
        self._agent_status = status

    @property
    def name(self) -> str:
        return self.__class__.__name__.lower()
