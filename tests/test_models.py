"""Tests for the CtrlAI validation models."""

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from validation.models import CtrlAI, CtrlAIGroup


def test_valid_ctrlai():
    data = {
        "@context": ["https://schema.org/", "https://ctrlai.com/schema/"],
        "@type": "CtrlAI",
        "id": f"urn:uuid:{uuid4()}",
        "userId": f"urn:uuid:{uuid4()}",
        "ctrlaiGroupId": f"urn:uuid:{uuid4()}",
        "type": "preference:dietary",
        "value": "vegetarian",
        "source": "userInput",
        "confidence": 0.9,
        "dateCreated": datetime.now(timezone.utc).isoformat(),
        "dateModified": datetime.now(timezone.utc).isoformat(),
        "expires": None,
        "validFrom": None,
        "scope": "personal",
        "keywords": ["food", "vegetarian"],
        "relatedEntities": [],
        "embedding": [0.1, 0.2, 0.3],
    }
    ctrlai = CtrlAI(**data)
    assert ctrlai.type == "preference:dietary"
    assert ctrlai.value == "vegetarian"
    assert ctrlai.confidence == 0.9


def test_invalid_type():
    data = {
        "type": "invalid:type",
        "value": "test",
        "source": "userInput",
        "confidence": 0.9,
        "scope": "personal",
        "id": "123e4567-e89b-12d3-a456-426614174000",
        "user_id": "123e4567-e89b-12d3-a456-426614174001",
        "ctrlai_group_id": "123e4567-e89b-12d3-a456-426614174002",
        "dateCreated": datetime.now(),
        "dateModified": datetime.now(),
        "embedding": [0.1, 0.2, 0.3],
    }
    with pytest.raises(ValidationError):
        CtrlAI(**data)


def test_ctrlai_group_creation():
    """Test creating a CtrlAIGroup instance."""
    group_data = {
        "id": UUID("a1b2c3d4-e5f6-7890-1234-567890abcdef"),
        "name": "Test Group",
        "user_id": UUID("f0e9d8c7-b6a5-4321-fedc-ba9876543210"),
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
    }
    group = CtrlAIGroup(**group_data)
    assert group.name == "Test Group"
