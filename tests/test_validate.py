from datetime import datetime, timezone
from uuid import uuid4

import pytest
from jsonschema import ValidationError as JsonSchemaValidationError
from pydantic import ValidationError as PydanticValidationError

from ctrlai_core.validate import (
    validate_ctrlai_embedding,
    validate_ctrlai_group,
    validate_ctrlai_json,
    validate_ctrlai_pydantic,
)


@pytest.fixture
def valid_ctrlai_data():
    return {
        "@context": ["https://schema.org/", "https://ctrlai.com/schema/"],
        "@type": "CtrlAI",
        "id": f"urn:uuid:{uuid4()}",
        "userId": f"urn:uuid:{uuid4()}",
        "ctrlaiGroupId": f"urn:uuid:{uuid4()}",
        "type": "preference:dietary",
        "value": {"@type": "DietaryRestriction", "name": "Vegetarian"},
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


@pytest.fixture
def invalid_ctrlai_data(valid_ctrlai_data):
    """Create invalid data by omitting required fields"""
    invalid_data = valid_ctrlai_data.copy()
    del invalid_data["confidence"]
    return invalid_data


def test_validate_ctrlai_json_valid(valid_ctrlai_data):
    """Test validation of valid CtrlAI JSON data."""
    try:
        validate_ctrlai_json(valid_ctrlai_data)
    except JsonSchemaValidationError as e:
        pytest.fail(f"Valid data raised ValidationError: {e}")


def test_validate_ctrlai_json_invalid(invalid_ctrlai_data):
    """Test validation of invalid CtrlAI JSON data."""
    with pytest.raises(JsonSchemaValidationError):
        validate_ctrlai_json(invalid_ctrlai_data)


def test_validate_ctrlai_pydantic_valid(valid_ctrlai_data):
    """Test Pydantic validation of valid CtrlAI data."""
    try:
        ctrlai = validate_ctrlai_pydantic(valid_ctrlai_data)
        assert ctrlai.type == "preference:dietary"
        assert ctrlai.confidence == 0.9
    except Exception as e:
        pytest.fail(f"Valid data raised error: {e}")


def test_validate_ctrlai_pydantic_invalid(invalid_ctrlai_data):
    """Test Pydantic validation of invalid CtrlAI data."""
    with pytest.raises(PydanticValidationError) as exc_info:
        validate_ctrlai_pydantic(invalid_ctrlai_data)
    assert "confidence" in str(exc_info.value)


def test_validate_ctrlai_embedding():
    """Test embedding generation (currently not implemented)."""
    with pytest.raises(NotImplementedError):
        validate_ctrlai_embedding("test value")


def test_validate_ctrlai_group():
    """Test group validation (currently not implemented)."""
    group_data = {
        "id": f"urn:uuid:{uuid4()}",
        "name": "Personal Preferences",
        "user_id": f"urn:uuid:{uuid4()}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with pytest.raises(NotImplementedError):
        validate_ctrlai_group(group_data)
