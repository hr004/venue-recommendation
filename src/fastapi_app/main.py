import logging

import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StartletteHTTPException

from bot_information_retrieval.config import config
from bot_information_retrieval.fastapi_app.exception_handler import (
    rfc9457_exception_handler,
)
from bot_information_retrieval.fastapi_app.file_router import file_router
from bot_information_retrieval.fastapi_app.http_exception import HTTPError
from bot_information_retrieval.fastapi_app.openapi_tags import openapi_tags
from bot_information_retrieval.fastapi_app.status_router import router as status_router
from bot_information_retrieval.internal.telemetry.metrics import (
    HttpServerHandlingSecondsMiddleware,
    MetricsHelper,
)

logger = logging.getLogger(__name__)


servers_map = {
    "local": [
        {"url": "http://localhost:8080"},
        {"url": "https://bot-information-retrieval.dev.hootdev.com"},
    ],
    "dev": [{"url": "https://bot-information-retrieval.dev.hootdev.com"}],
    "staging": [{"url": "https://bot-information-retrieval.staging.hootops.com"}],
    "production": [{"url": "https://bot-information-retrieval.prod.hootops.com"}],
}


def create_app(metrics=None) -> FastAPI:
    app = FastAPI(
        title=config.logger.service_name,
        servers=servers_map.get(config.environment, []),
        openapi_tags=openapi_tags,
        openapi_url="/api/spec",
        docs_url="/api/swagger",
        redoc_url="/api/redoc",
        root_path_in_servers=False,
    )
    app.openapi_version = "3.0.0"

    if metrics is None:
        metrics = MetricsHelper(config.metrics.host, config.metrics.port)

    app.extra["metrics"] = metrics

    app.add_middleware(HttpServerHandlingSecondsMiddleware, metrics=metrics)

    app.include_router(file_router)
    app.include_router(status_router)

    app.add_exception_handler(RequestValidationError, rfc9457_exception_handler)
    app.add_exception_handler(StartletteHTTPException, rfc9457_exception_handler)
    app.add_exception_handler(Exception, rfc9457_exception_handler)
    app.add_exception_handler(HTTPError, rfc9457_exception_handler)
    return app


def serve(metrics=None):
    app = create_app(metrics)
    uvicorn.run(app, host=config.server.host, port=config.server.port, log_config=None)


if __name__ == "__main__":
    serve()
