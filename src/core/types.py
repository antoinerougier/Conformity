"""Shared constrained types and the runtime input-validation decorator.

Every public function that receives data from another layer (UI, CLI, LLM output, files)
is decorated with :func:`validated`, so invalid arguments fail fast with a
:class:`pydantic.ValidationError` instead of propagating silently through the pipeline.

Purely internal hot-path helpers (tokenizers, parser state machine) are intentionally *not*
decorated: they only receive data that has already been validated at the boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, TypeVar

from pydantic import ConfigDict, Field, StringConstraints, validate_call

F = TypeVar("F", bound=Callable[..., Any])

#: Non-blank string, surrounding whitespace stripped.
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
#: Strictly positive integer.
PositiveInt = Annotated[int, Field(gt=0)]
#: Non-negative integer.
NonNegativeInt = Annotated[int, Field(ge=0)]
#: Score or probability in the closed interval [0, 1].
UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]

_VALIDATION_CONFIG = ConfigDict(arbitrary_types_allowed=True, strict=False)


def validated(func: F) -> F:
    """Validate call arguments against the function's type hints (Pydantic v2).

    ``arbitrary_types_allowed`` lets services (vector store, LLM client...) be passed and
    checked with ``isinstance`` while plain data is fully validated and coerced.
    """
    return validate_call(config=_VALIDATION_CONFIG)(func)
