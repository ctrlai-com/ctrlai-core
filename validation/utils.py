"""
Utility functions for the CtrlAI validation package.
"""

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from .constants import CtrlAIScope, CtrlAISource, CtrlAIType


def generate_ctrlai_id() -> str:
    """Generate a new CtrlAI UUID in URN format."""
    return f"urn:uuid:{uuid4()}"


def create_ctrlai_base(
    type: CtrlAIType,
    value: Any,
    source: CtrlAISource,
    confidence: float,
    scope: CtrlAIScope,
    keywords: Optional[list] = None,
    related_entities: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Create a base CtrlAI dictionary with required fields.

    Args:
        type: The CtrlAI type
        value: The content value
        source: The source of the information
        confidence: Confidence level (0.0 to 1.0)
        scope: The scope of the information
        keywords: Optional list of keywords
        related_entities: Optional list of related entity URIs

    Returns:
        Dict containing the base CtrlAI structure
    """
    now = datetime.utcnow().isoformat() + "Z"

    return {
        "@context": ["https://schema.org/", "https://ctrlai.com/schema/"],
        "@type": "CtrlAI",
        "id": generate_ctrlai_id(),
        "type": type.value,
        "value": value,
        "source": source.value,
        "confidence": confidence,
        "dateCreated": now,
        "dateModified": now,
        "scope": scope.value,
        "keywords": keywords or [],
        "relatedEntities": related_entities or [],
    }


def is_valid_uuid(uuid_str: str) -> bool:
    """Check if a string is a valid UUID."""
    try:
        UUID(uuid_str)
        return True
    except ValueError:
        return False
