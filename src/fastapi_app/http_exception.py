from enum import Enum
from http import HTTPStatus

from starlette.exceptions import HTTPException as StarletteHTTPException

HTTPErrorType = Enum(
    "HTTPErrorType",
    [
        "UnprocessableEntityError",
        "HTTPError",
    ],
)


class HTTPError(StarletteHTTPException):
    def __init__(
        self,
        status_code: int,
        title: str,
        error_type: HTTPErrorType,
        detail: str,
    ) -> None:
        super().__init__(status_code=status_code, detail=detail, headers=None)
        self.title = title
        self.error_type = error_type

    def getExtentionAttributes(self) -> dict:
        return {}


class UnprocessableEntityHTTPError(HTTPError):
    def __init__(
        self,
        detail: str,
        errors: dict = {},
    ) -> None:
        super().__init__(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            title=HTTPStatus.UNPROCESSABLE_ENTITY.phrase,
            error_type=HTTPErrorType.UnprocessableEntityError,
            detail=detail,
        )
        self.errors = errors

    def getExtentionAttributes(self) -> dict:
        return {"errors": self.errors}
