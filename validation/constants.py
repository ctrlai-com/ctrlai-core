"""
Constants used throughout the CtrlAI validation package.
"""

from enum import Enum
from typing import Dict


class CtrlAIType(str, Enum):
    """Enumeration of valid CtrlAI types."""

    # Preferences
    DIETARY = "preference:dietary"
    COMMUNICATION = "preference:communication_style"
    WRITING = "preference:writing_style"
    PERSONALITY = "preference:personality"
    PERSONAL_INTEREST = "preference:personal_interest"

    # Context
    PROJECT = "context:project"
    LOCATION = "context:location"
    TIME_SENSITIVE = "context:time_sensitive"
    GOAL = "context:goal"

    # Knowledge
    COMPANY_FACT = "knowledge:company_fact"
    PRODUCT_SPEC = "knowledge:product_spec"
    COMPETITOR_INFO = "knowledge:competitor_info"
    PERSONAL_FACT = "knowledge:personal_fact"
    CALENDAR = "knowledge:calendar"
    SKILLS = "knowledge:skills"
    EXTERNAL = "knowledge:external"


class CtrlAIScope(str, Enum):
    """Enumeration of valid CtrlAI scopes."""

    PERSONAL = "personal"
    PROJECT = "project:"
    DEPARTMENT = "department:"
    COMPANY = "company"


class CtrlAISource(str, Enum):
    """Enumeration of common CtrlAI sources."""

    USER_INPUT = "userInput"
    SYSTEM = "system"
    COMPANY_WIKI = "companyWiki"
    PROJECT_MANAGEMENT = "projectManagementSystem"
    CALENDAR = "calendar"
    EXTERNAL_API = "externalApi"


# Schema-related constants
SCHEMA_VERSION = "1.0"
REQUIRED_CONTEXT = ["https://schema.org/", "https://ctrlai.com/schema/"]

# Validation-related constants
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0
EMBEDDING_DIMENSION = 1536  # Example for OpenAI's text-embedding-ada-002

# Define valid types for CtrlAI entries
VALID_TYPES: Dict[str, str] = {
    "preference:dietary": "Dietary preferences",
    "preference:communication_style": "Communication style preferences",
    "preference:writing_style": "Writing style preferences",
    "preference:personality": "Personality traits",
    "preference:personal_interest": "Personal interests",
    "context:project": "Project context",
    "context:location": "Location context",
    "context:time_sensitive": "Time-sensitive context",
    "context:goal": "Goal context",
    "knowledge:company_fact": "Company facts",
    "knowledge:product_spec": "Product specifications",
    "knowledge:competitor_info": "Competitor information",
    "knowledge:personal_fact": "Personal facts",
    "knowledge:calendar": "Calendar information",
    "knowledge:skills": "Skills information",
    "knowledge:external": "External knowledge",
}

# Define valid scopes
VALID_SCOPES = ["personal", "project:", "department:", "company"]
