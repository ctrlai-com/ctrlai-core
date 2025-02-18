# ctrlai_core/models.py
from datetime import datetime
from typing import (
    List,
    Optional,
    Union
)
from uuid import UUID

from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    field_validator
)


class CtrlAIGroupBase(BaseModel):
    name: str = Field(
        ...,
        description="Group name"
    )


class CtrlAIGroup(CtrlAIGroupBase):
    id: UUID = Field(
        ..., 
        description="Group ID"
    )
    user_id: UUID = Field(
        ..., 
        description="Owner ID"
    )
    created_at: datetime = Field(
        ..., 
        description="Created at"
    )
    updated_at: datetime = Field(
        ..., 
        description="Updated at"
    )

    model_config = ConfigDict(from_attributes=True)


class CtrlAIBase(BaseModel):
    type: str = Field(..., description="Type")
    value: Union[str, dict] = Field(..., description="Value")
    source: str = Field(..., description="Source")
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Confidence"
    )
    expires: Optional[str] = Field(None, description="Expires")
    validFrom: Optional[str] = Field(None, description="Valid from")
    scope: str = Field(..., description="Scope")
    keywords: List[str] = Field(
        default_factory=list,
        description="Keywords"
    )
    relatedEntities: List[AnyUrl] = Field(
        default_factory=list,
        description="Links"
    )


class CtrlAI(CtrlAIBase):
    context: List[AnyUrl] = Field(
        ...,
        alias="@context"
    )
    type_: str = Field(
        ...,
        alias="@type"
    )
    id: str = Field(
        ...,
        pattern=(
            "^urn:uuid:[0-9a-f]{8}-[0-9a-f]{4}-"
            "[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        )
    )
    userId: str = Field(
        ...,
        pattern=(
            "^urn:uuid:[0-9a-f]{8}-[0-9a-f]{4}-"
            "[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        )
    )
    ctrlaiGroupId: str = Field(
        ...,
        pattern=(
            "^urn:uuid:[0-9a-f]{8}-[0-9a-f]{4}-"
            "[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        )
    )
    dateCreated: str = Field(
        ...,
        description="Creation timestamp"
    )
    dateModified: str = Field(
        ...,
        description="Modification timestamp"
    )
    embedding: List[float] = Field(
        ...,
        description="Vector embedding"
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={datetime: lambda v: v.isoformat()}
    )

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        valid_types = [
            "preference:dietary",
            "preference:communication_style",
            "preference:writing_style",
            "preference:personality",
            "preference:personal_interest",
            "context:project",
            "context:location",
            "context:time_sensitive",
            "context:goal",
            "knowledge:company_fact",
            "knowledge:product_spec",
            "knowledge:competitor_info",
            "knowledge:personal_fact",
            "knowledge:calendar",
            "knowledge:skills",
            "knowledge:external",
        ]
        if v not in valid_types:
            raise ValueError(
                f'Invalid type: {v}. Must be one of: {", ".join(valid_types)}'
            )
        return v

    @field_validator("scope")
    @classmethod
    def validate_scope(cls, v: str) -> str:
        valid_scopes = ["personal", "project:", "department:", "company"]
        if not any(v.startswith(scope) for scope in valid_scopes):
            raise ValueError(f"Invalid scope: {v}")
        return v


# Version info as module-level constants
VERSION = "1.0.0"
SCHEMA_VERSION = "1.0"
API_VERSION = "v1"
