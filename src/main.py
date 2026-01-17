import logging
import sys

import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv
from src.config import config
from src.fastapi_app.file_router import file_router
from src.fastapi_app.status_router import router as status_router

logger = logging.getLogger(__name__)

load_dotenv()


def setup_logging():
    """Configure logging for the application."""
    log_level = getattr(logging, config.logger.level.upper(), logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # Override any existing configuration
    )

    # Set log level for uvicorn
    logging.getLogger("uvicorn").setLevel(log_level)
    logging.getLogger("uvicorn.access").setLevel(log_level)
    logging.getLogger("uvicorn.error").setLevel(log_level)

    # Set log level for application loggers
    logging.getLogger("src").setLevel(log_level)

    logger.info(f"Logging configured at {config.logger.level} level")


# Initialize logging before creating the app
setup_logging()


def create_app() -> FastAPI:
    app = FastAPI(
        title=config.logger.service_name,
        servers=[{"url": config.server.get_server_url()}],
        docs_url="/api/docs",
        root_path_in_servers=False,
    )

    app.include_router(file_router)
    app.include_router(status_router)
    # TODO: add exception handlers
    return app


def serve():
    app = create_app()
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.logger.level.lower(),  # Pass log level to uvicorn
        access_log=True,  # Enable access logs
    )


if __name__ == "__main__":
    serve()
