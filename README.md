# Ctrl AI Core: Protocol Specification

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository defines the core protocol for Ctrl AI, an open standard for personalized AI knowledge. Ctrl AI empowers individuals and organizations to control their AI interactions by providing a structured and interoperable way to manage the information that AI agents use.

## What is Ctrl AI?

Ctrl AI is a system for creating and managing *Ctrl AIs* - structured pieces of information that provide context, preferences, and knowledge to AI agents. This allows for:

* **Enhanced AI Performance:** More relevant, accurate, and personalized AI responses
* **Increased Efficiency:** Reduced need to repeatedly provide context
* **User Control:** Fine-grained control over what information AI agents can access
* **Interoperability:** A standardized way for different AI systems to share and use knowledge
* **Collaboration:** Sharing of knowledge within teams and organizations

## Core Components

This repository contains the following key components:

* **[Ctrl AI Blueprint](docs/blueprint.mdx):** A comprehensive overview of the Ctrl AI protocol, its goals, and its architecture
* **[Custom Vocabulary Definition](docs/vocabulary.mdx):** A formal definition of the Ctrl AI-specific types and properties
* **[JSON Schema](docs/schema.json):** A machine-readable schema that defines the structure of a Ctrl AI
* **[API Specification](docs/api_specification.mdx):** A high-level overview of the API principles
* **[Examples](examples/):** Example JSON-LD documents conforming to the Ctrl AI protocol
* **[Validation Package](validation/):** Python package with Pydantic models and validation functions

## Installation

```bash
# Using pip
pip install ctrlai-core

# Using poetry
poetry add ctrlai-core

# Development installation
git clone https://github.com/ctrlai-com/ctrlai-core.git
cd ctrlai-core
poetry install
```

## Quick Start

```python
from ctrlai_core.validation import CtrlAI
from datetime import datetime

# Create and validate a Ctrl AI
ctrlai_data = {
    "@context": [
        "https://schema.org/",
        "https://ctrlai.com/schema/"
    ],
    "@type": "CtrlAI",
    "id": "urn:uuid:a1b2c3d4-e5f6-7890-1234-567890abcdef",
    "type": "preference:dietary",
    "value": {
        "@type": "DietaryRestriction",
        "name": "Vegetarian"
    },
    "source": "userInput",
    "confidence": 0.9,
    "scope": "personal"
}

try:
    ctrlai = CtrlAI(**ctrlai_data)
    print("✓ Ctrl AI is valid!")
except ValidationError as e:
    print(f"✗ Validation Error: {e}")
```

## Documentation

For detailed documentation, visit our [Documentation Site](https://ctrlai-com.github.io/ctrlai-core/).

* [Protocol Overview](docs/blueprint.mdx)
* [API Reference](docs/api_specification.mdx)
* [Examples](examples/)
* [Contributing Guide](CONTRIBUTING.md)

## Development

```bash
# Install development dependencies
poetry install

# Run tests
poetry run pytest

# Run linting
poetry run black .
poetry run isort .
poetry run flake8 .

# Run type checking
poetry run mypy validation
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

* [GitHub Issues](https://github.com/ctrlai-com/ctrlai-core/issues)
* [Documentation](https://ctrlai-com.github.io/ctrlai-core/)
