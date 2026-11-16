"""Domain exception hierarchy.

Callers can catch :class:`ComplianceAuditorError` to handle every expected failure of the
application, while unexpected errors (bugs) still surface as standard exceptions.
"""

from __future__ import annotations


class ComplianceAuditorError(Exception):
    """Base class for all expected application errors."""


class UnsupportedDocumentError(ComplianceAuditorError, ValueError):
    """The uploaded document has an unsupported format, is corrupted or is empty."""


class CorpusIntegrityError(ComplianceAuditorError, ValueError):
    """The legal corpus is incomplete or inconsistent (missing articles, duplicate ids...)."""


class StructuredOutputError(ComplianceAuditorError, RuntimeError):
    """The LLM did not return output matching the requested schema after retries."""


class AuditExecutionError(ComplianceAuditorError, RuntimeError):
    """The audit graph finished without producing a report."""
