import logging
from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile, status

from venue_rec.config import DEFAULT_VECTORSTORE_TYPE
from venue_rec.datamodel.annotations import (
    BotIdAnnotation,
    CursorIdAnnotation,
    FileIdAnnotation,
    KnowledgeFileTypeAnnotation,
    OrgIdAnnotation,
)
from venue_rec.datamodel.api.request import SearchRequest
from venue_rec.datamodel.api.response import (
    ListKnowledgeFilesResponse,
    SearchResponse,
    SearchResponseAllChunks,
    UploadResponse,
)
from venue_rec.datamodel.model import FileConfig, MimeType, RagConfig
from venue_rec.engine import BotRetrievalAgent
from venue_rec.fastapi_app.exception_handler import create_error_dto
from venue_rec.fastapi_app.http_exception import (
    InternalHTTPError,
    UnprocessableEntityHTTPError,
    UnsupportedFileTypeHTTPError,
)
from venue_rec.fastapi_app.openapi_tags import bot_config_file_tag

supported_file_MIME_type = [
    MimeType.pdf,
    MimeType.plain,
]

logger = logging.getLogger(__name__)

file_router = APIRouter(
    prefix="/internal/v1/organizations/{orgId}/chatbot-config-ids/{chatbotConfigId}",
    tags=[bot_config_file_tag],
)
upload_size_limit_KB = 256


@file_router.post(
    "/knowledge-files",
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {
            "description": "File uploaded successfully",
        },
        415: {
            "description": "Invalid file type",
            "content": create_error_dto(
                UnsupportedFileTypeHTTPError("Unsupported file type")
            ),
        },
    },
    response_model=UploadResponse,
)
async def upload_knowledge_file(
    org_id: OrgIdAnnotation,
    chatbot_config_id: BotIdAnnotation,
    file: Annotated[
        UploadFile,
        File(
            description=f"The knowledge file to upload. Must be of mime type: {supported_file_MIME_type}"
        ),
    ],
    knowledge_file_type: (
        KnowledgeFileTypeAnnotation | None
    ) = KnowledgeFileTypeAnnotation.FAQ,
    file_id: Annotated[str, Form(..., alias="fileId")] | None | None = None,
) -> UploadResponse:
    """Uploads a file to index in our custom rag."""
    logger.debug(f"Upload file with name: {file.filename}")

    if not (file.content_type in supported_file_MIME_type):
        raise UnsupportedFileTypeHTTPError("Unsupported file type")

    file_data = await file.read()

    rag_config = RagConfig(
        file=FileConfig(
            file_paths=[],
            enabled=True,
            vectorstore_type=DEFAULT_VECTORSTORE_TYPE,
            metadata={
                "source": file.filename,
                "chatbot_config_id": chatbot_config_id,
                "org_id": org_id,
                "knowledge_file_type": knowledge_file_type,
                "file_id": file_id,
            },
        ),
    )
    try:
        bot_retrieval = BotRetrievalAgent(config=rag_config)
        indexed_knowledge_file = await bot_retrieval.index(
            file_contents=file_data,
            mime_type=file.content_type,
        )
        return UploadResponse(
            chatbot_config_id=chatbot_config_id,
            org_id=org_id,
            knowledge_file_id=indexed_knowledge_file.file_id,
            knowledge_file_name=indexed_knowledge_file.filename,
        )

    except Exception as e:
        raise InternalHTTPError(f"Error uploading file: {e}")


@file_router.get(
    "/knowledge-files/{fileId}/chunks",
    status_code=status.HTTP_200_OK,
    responses={
        422: {
            "description": "Unprocessable Entity",
            "content": create_error_dto(
                UnprocessableEntityHTTPError("Unprocessable Entity")
            ),
        },
    },
    response_model=SearchResponseAllChunks,
)
async def get_all_documents(
    org_id: OrgIdAnnotation,
    chatbot_config_id: BotIdAnnotation,
    file_id: FileIdAnnotation,
    cursor_id: CursorIdAnnotation = None,
) -> dict:
    logger.debug("Get files")

    rag_config = RagConfig(
        file=FileConfig(
            file_paths=[],
            enabled=True,
            vectorstore_type=DEFAULT_VECTORSTORE_TYPE,
            metadata={
                "chatbot_config_id": chatbot_config_id,
                "org_id": org_id,
                "file_id": file_id,
            },
        ),
    )
    bot_retrieval = BotRetrievalAgent(config=rag_config)
    knowledge_file_list, next_search_after = await bot_retrieval.retrieve_all_documents(
        search_after=cursor_id
    )
    return {
        "documents": knowledge_file_list,
        "cursor_ids": next_search_after,
    }


@file_router.post(
    "/searchRequests",
    status_code=200,
    responses={
        422: {
            "description": "Invalid search request",
            "content": create_error_dto(
                UnprocessableEntityHTTPError("Invalid search request")
            ),
        },
    },
    response_model=SearchResponse,
)
async def get_nearest_documents(
    org_id: OrgIdAnnotation,
    chatbot_config_id: BotIdAnnotation,
    request: SearchRequest,
) -> dict:
    logger.debug("Search nearest documents")

    rag_config = RagConfig(
        file=FileConfig(
            file_paths=[],
            enabled=True,
            vectorstore_type="opensearch",
            metadata={
                "chatbot_config_id": chatbot_config_id,
                "org_id": org_id,
                "knowledge_file_type": request.knowledge_file_type,
            },
        ),
    )
    try:
        bot_retrieval = BotRetrievalAgent(config=rag_config)
        documents = await bot_retrieval.retrieve(request.query, size=request.k)
        return {"documents": [doc.dict() for doc in documents]}
    except Exception as e:
        raise InternalHTTPError(f"Error searching documents: {e}")


@file_router.delete(
    "/knowledge-files/{fileId}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        422: {
            "description": "RequestValidationError",
            "content": create_error_dto(
                UnprocessableEntityHTTPError("RequestValidationError")
            ),
        }
    },
)
async def delete_file(
    org_id: OrgIdAnnotation,
    chatbot_config_id: BotIdAnnotation,
    file_id: FileIdAnnotation,
) -> None:
    logger.debug(f"Delete file with id: {file_id}")

    rag_config = RagConfig(
        file=FileConfig(
            file_paths=[],
            enabled=True,
            vectorstore_type="opensearch",
            metadata={
                "chatbot_config_id": chatbot_config_id,
                "org_id": org_id,
            },
        ),
    )
    bot_retrieval = BotRetrievalAgent(config=rag_config)
    try:
        _ = await bot_retrieval.delete(file_id)
    except Exception as e:
        raise InternalHTTPError(f"Error deleting file: {e}")


@file_router.get(
    "/knowledge-files",
    status_code=200,
    responses={
        422: {
            "description": "RequestValidationError",
            "content": create_error_dto(
                UnprocessableEntityHTTPError("RequestValidationError")
            ),
        },
    },
    description="List all files",
    response_model=ListKnowledgeFilesResponse,
)
async def list_files(
    org_id: OrgIdAnnotation,
    chatbot_config_id: BotIdAnnotation,
) -> dict:
    logger.debug("List files")

    rag_config = RagConfig(
        file=FileConfig(
            file_paths=[],
            enabled=True,
            vectorstore_type=DEFAULT_VECTORSTORE_TYPE,
            metadata={"chatbot_config_id": chatbot_config_id, "org_id": org_id},
        ),
    )
    bot_retrieval = BotRetrievalAgent(config=rag_config)
    try:
        knowledge_file_list = await bot_retrieval.list_files()
        return {
            "knowledge_files": knowledge_file_list,
        }
    except Exception as e:
        raise InternalHTTPError(f"Error listing files: {e}")
