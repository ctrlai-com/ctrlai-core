# ctrlai_core/validation/validate.py

"""
CtrlAI Validation Module

This module provides validation functions for CtrlAI data structures,
including JSON Schema validation, Pydantic model validation,
embedding generation, and group validation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Union

from jsonschema import ValidationError, validate

from .models import CtrlAI, CtrlAIGroup

# Load the JSON Schema
SCHEMA_PATH = Path(__file__).parent.parent / "docs" / "schema.json"
with open(SCHEMA_PATH, "r") as f:
    schema = json.load(f)


def validate_ctrlai_json(data: Dict[str, Any]) -> None:
    """
    Validates a JSON object against the Ctrl AI JSON Schema.

    Args:
        data: Dictionary containing the CtrlAI data to validate.

    Raises:
        jsonschema.ValidationError: If invalid per schema.
    """
    validate(instance=data, schema=schema)


def validate_ctrlai_pydantic(data: Dict[str, Any]) -> CtrlAI:
    """Validates a JSON object against the Ctrl AI Pydantic model.

    Args:
        data: Dictionary containing the CtrlAI data to validate.

    Returns:
        CtrlAI: A validated CtrlAI object.

    Raises:
        pydantic.ValidationError: If validation fails.
    """
    return CtrlAI(**data)


def validate_ctrlai_embedding(
    value: Union[str, Dict[str, Any]]
) -> List[float]:
    """
    Generates an embedding for a Ctrl AI value.

    Args:
        value: The value to generate an embedding for. Can be either
            a string or a structured dictionary.

    Returns:
        List[float]: The generated embedding vector.

    Raises:
        NotImplementedError: Currently not implemented.
    """
    # TODO: Implement embedding generation
    raise NotImplementedError("Embedding generation not implemented")


def validate_ctrlai_group(data: Dict[str, Any]) -> CtrlAIGroup:
    """
    Validates a Ctrl AI Group against the schema.

    Args:
        data: Dictionary containing the CtrlAI Group data to validate.

    Returns:
        CtrlAIGroup: A validated CtrlAIGroup object.

    Raises:
        ValidationError: If the data is invalid.
        NotImplementedError: Currently not implemented.
    """
    # TODO: Implement group validation
    raise NotImplementedError("Group validation not implemented")


# Example Usage:
if __name__ == "__main__":
    # Example valid Ctrl AI data
    valid_data: Dict[str, Any] = {
        "@context": ["https://schema.org/", "https://ctrlai.com/schema/"],
        "@type": "CtrlAI",
        "id": "urn:uuid:a1b2c3d4-e5f6-7890-1234-567890abcdef",
        "userId": "urn:uuid:f0e9d8c7-b6a5-4321-fedc-ba9876543210",
        "ctrlaiGroupId": "urn:uuid:01234567-89ab-cdef-0123-456789abcdef",
        "type": "preference:dietary",
        "value": {"@type": "DietaryRestriction", "name": "Vegetarian"},
        "source": "userInput",
        "confidence": 0.9,
        "dateCreated": "2024-01-28T14:30:00Z",
        "dateModified": "2024-01-28T14:30:00Z",
        "expires": None,
        "validFrom": None,
        "scope": "personal",
        "keywords": ["food", "vegetarian"],
        "relatedEntities": [],
        "embedding": [0.1, 0.2, 0.3],
    }

    # Example invalid Ctrl AI data
    invalid_data: Dict[str, Any] = {
        "@context": ["https://schema.org/", "https://ctrlai.com/schema/"],
        "@type": "CtrlAI",
        "id": "urn:uuid:a1b2c3d4-e5f6-7890-1234-567890abcdef",
        "userId": "urn:uuid:f0e9d8c7-b6a5-4321-fedc-ba9876543210",
        "ctrlaiGroupId": "urn:uuid:01234567-89ab-cdef-0123-456789abcdef",
        "type": "preference:dietary",
        "value": {"@type": "DietaryRestriction", "name": "Vegetarian"},
        "source": "userInput",
        # Missing "confidence"
        "dateCreated": "2025-02-17T14:30:00Z",
        "dateModified": "2025-02-17T14:30:00Z",
        "expires": None,
        "validFrom": None,
        "scope": "personal",
        "keywords": ["food", "vegetarian"],
        "relatedEntities": [],
    }

    # Test different value types
    value_types: Dict[str, Dict[str, Any]] = {
        "string_value": {"@type": "CtrlAI", "value": "Simple string value"},
        "dict_value": {
            "@type": "CtrlAI",
            "value": {"@type": "DietaryRestriction", "name": "Vegetarian"},
        },
        "complex_value": {
            "@type": "CtrlAI",
            "value": {
                "@type": "ProjectContext",
                "name": "Project Alpha",
                "deadline": "2024-03-15",
                "team": ["Alice", "Bob"],
            },
        },
    }

    # Test group validation
    group_data: Dict[str, Any] = {
        "id": "urn:uuid:abcdef01-2345-6789-abcd-ef0123456789",
        "name": "Personal Preferences",
        "user_id": "urn:uuid:fedcba98-7654-3210-fedc-ba9876543210",
        "created_at": "2024-01-28T10:00:00Z",
        "updated_at": "2024-01-28T10:00:00Z",
    }

    # Test embedding generation
    test_values = [
        "I prefer vegetarian food",
        "Working on Project Alpha",
        "Company revenue for Q1 2024",
    ]

    print("\n=== Testing Different Value Types ===")
    for test_name, test_data in value_types.items():
        try:
            validate_ctrlai_json(test_data)
            print(f"✓ {test_name}: Valid")
        except ValidationError as e:
            print(f"✗ {test_name}: Invalid - {e}")

    print("\n=== Testing Group Validation ===")
    try:
        validate_ctrlai_group(group_data)
        print("✓ Group validation: Valid")
    except ValidationError as e:
        print(f"✗ Group validation: Invalid - {e}")
    except NotImplementedError:
        print("! Group validation: Not implemented yet")

    print("\n=== Testing Embedding Generation ===")
    for value in test_values:
        try:
            embedding = validate_ctrlai_embedding(value)
            print(f"✓ Generated embedding for: {value[:30]}...")
        except NotImplementedError:
            print("! Embedding generation: Not implemented yet")
        except Exception as e:
            print(f"✗ Failed to generate embedding for: {value[:30]}... - {e}")

    # Test the original example data
    print("\n=== Testing Example Data ===")
    try:
        validate_ctrlai_json(valid_data)
        print("✓ Valid data (jsonschema): OK")
    except ValidationError:
        print("✗ Invalid data (jsonschema)")

    try:
        validate_ctrlai_json(invalid_data)
        print("✗ Invalid data test passed (caught expected error)")
    except ValidationError:
        print("✓ Invalid data correctly rejected")

    # Test Pydantic validation
    print("\n=== Testing Pydantic Validation ===")
    try:
        ctrlai_object = validate_ctrlai_pydantic(valid_data)
        print("✓ Valid data (Pydantic): OK")
    except ValidationError as e:
        print(f"✗ Invalid data (Pydantic): {e}")

    try:
        validate_ctrlai_pydantic(invalid_data)
        print("✗ Invalid data test failed (should have raised error)")
    except ValidationError:
        print("✓ Invalid data correctly rejected by Pydantic")
