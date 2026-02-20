import logging
from http import HTTPStatus
from typing import Generic, TypeVar

from kagglehub.exceptions import ColabHTTPError, KaggleApiHTTPError, UnauthenticatedError

from kagglehub.handle import CompetitionHandle, DatasetHandle, ModelHandle, NotebookHandle, ResourceHandle
from kagglehub.resolver import Resolver

T = TypeVar("T", bound=ResourceHandle)
logger = logging.getLogger(__name__)


class MultiImplRegistry(Generic[T]):
    """Utility class to inject multiple implementations of class.

    Each implementation must implement __call__ and is_supported with the same set of arguments. The registered
    implementations "is_supported" methods are called in reverse order under which they are registered. The first
    to return true is then invoked via __call__ and the result returned.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._impls: list[Resolver[T]] = []

    def add_implementation(self, impl: Resolver[T]) -> None:
        self._impls.append(impl)

    def __call__(self, *args, **kwargs) -> tuple[str, int | None]:  # noqa: ANN002, ANN003
        fails = []
        last_auth_exception: Exception | None = None
        for impl in reversed(self._impls):
            impl_name = type(impl).__name__
            try:
                if not impl.is_supported(*args, **kwargs):
                    fails.append(impl_name)
                    continue
                return impl(*args, **kwargs)
            except Exception as exc:
                if _is_auth_failure(exc):
                    logger.warning(f"Authentication failed for resolver {impl_name}; trying next fallback.")
                    fails.append(impl_name)
                    last_auth_exception = exc
                    continue
                raise

        if last_auth_exception is not None:
            raise last_auth_exception

        msg = f"Missing implementation that supports: {self._name}(*{args!r}, **{kwargs!r}). Tried {fails!r}"
        raise RuntimeError(msg)


def _is_auth_failure(exc: Exception) -> bool:
    if isinstance(exc, UnauthenticatedError):
        return True

    if isinstance(exc, (KaggleApiHTTPError, ColabHTTPError)) and exc.response is not None:
        return exc.response.status_code in {HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN}

    return False


model_resolver = MultiImplRegistry[ModelHandle]("ModelResolver")
dataset_resolver = MultiImplRegistry[DatasetHandle]("DatasetResolver")
competition_resolver = MultiImplRegistry[CompetitionHandle]("CompetitionResolver")
notebook_output_resolver = MultiImplRegistry[NotebookHandle]("NotebookOutputResolver")
