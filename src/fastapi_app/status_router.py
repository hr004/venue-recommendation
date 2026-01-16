import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/status",
)


@router.get("/am-i-up", status_code=200)
async def status_up(request: Request):
    """Endpoint used to check if the service is running.

    Returns:
        Returns "OK", with status_code=200
    """
    logger.debug("Service is running")
    return JSONResponse(content={"message": "Service is running"})
