import json
import logging

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from starlette.exceptions import HTTPException as StartletteHTTPException

from venue_rec.fastapi_app.http_exception import (
    HTTPError,
    HTTPErrorType,
    InternalHTTPError,
    UnprocessableEntityHTTPError,
)


logger = logging.getLogger(__name__)


def create_error_dto(http_error: HTTPError):
    return {
        "application/problem+json": {
            "example": {
                "type": http_error.error_type.name,
                "title": http_error.title,
                "status": http_error.status_code,
                "detail": str(http_error.detail),
                "instance": "123456",
                **http_error.getExtentionAttributes(),
            }
        }
    }


async def rfc9457_exception_handler(request: Request, exc: Exception) -> Response:
    try:
        # Try to debug maximum recursion error https://hootsuite.atlassian.net/browse/CHAT-494
        exc_str = str(exc)
    except Exception as e:
        exc_str = f"Error occurred while converting exception to string: {e}"

    http_error: HTTPError = InternalHTTPError("Internal Error, see logs")
    if isinstance(exc, HTTPError):
        http_error = exc
    elif isinstance(exc, StartletteHTTPException):
        http_error = HTTPError(
            status_code=exc.status_code,
            error_type=HTTPErrorType.UnkownError,
            title=type(exc).__name__,
            detail=Exception(exc.detail),
        )
    elif isinstance(exc, RequestValidationError):
        http_error = UnprocessableEntityHTTPError(
            "Request validation error(s)", errors=exc.errors()
        )

    if http_error.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR:
        logger.error(
            f"Unhandled exception ({type(exc).__name__}) in FastAPI app: {exc_str}",
            exc_info=True,
        )

    # RFC9457 Problem Details for HTTP APIs
    # https://datatracker.ietf.org/doc/html/rfc9457#name-members-of-a-problem-detail
    return Response(
        status_code=http_error.status_code,
        content=json.dumps(
            {
                "type": http_error.error_type.name,
                "title": http_error.title,
                "status": http_error.status_code,
                "detail": str(http_error.detail),
                # "instance": str(get_request_id(request)),
                **http_error.getExtentionAttributes(),
            },
            default=lambda _o: "<not serializable>",  # pyright: ignore | "_o" is not accessed
        ),
        headers={"content-type": "application/problem+json"},
        media_type="application/problem+json",
    )
