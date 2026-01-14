import logging
from abc import abstractmethod

from fastapi import APIRouter, Request

from bot_information_retrieval.config import DEFAULT_VECTORSTORE_TYPE
from bot_information_retrieval.engine import VectorStoreFactory
from bot_information_retrieval.fastapi_app.http_exception import (
    ServiceUnavailableHTTPError,
)
from bot_information_retrieval.fastapi_app.openapi_tags import status_tag

logger = logging.getLogger(__name__)


class StatusHandler:
    @abstractmethod
    async def is_up(self) -> bool:
        pass


class OpensearchStatusHandler(StatusHandler):
    async def is_up(self) -> bool:
        try:
            store = VectorStoreFactory().create_vectorstore(DEFAULT_VECTORSTORE_TYPE)
            return await store.ping()
        except Exception as e:
            logger.exception("error when checking if OpenSearch is up", e)
            return False


router = APIRouter(
    prefix="/status",
)

opensearch_status_handler = OpensearchStatusHandler()


@router.get("/am-i-up", status_code=200, tags=[status_tag])
async def status_up(request: Request):
    """Endpoint used for the kubernetes liveness and readiness probes to determine if the service is running.

    Returns:
        Returns "OK", with status_code=200
    """
    logger.debug("Liveness Probe: OK")
    return "OK"


@router.get("/readiness", status_code=200, tags=[status_tag])
async def status_ready():
    """Endpoint used for the Kubernetes readiness probe to determine if the service is running.

    Checks the health of the ElasticSearch instance used by the service.

    Returns:
        Returns "OK" with status_code=200 if ElasticSearch is healthy.
        Raises ServiceUnavailableHTTPError if ElasticSearch is not healthy.
    """
    is_healthy = await opensearch_status_handler.is_up()
    logger.debug("Readiness Probe: OK")
    if not is_healthy:
        raise ServiceUnavailableHTTPError(detail="OpenSearch cluster is not healthy")
    return "OK"
