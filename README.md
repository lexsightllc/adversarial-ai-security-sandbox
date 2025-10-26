# AI Adversarial Sandbox

This repository hosts the professionalized version of the AI Adversarial Sandbox, evolving from a simple Flask application into a robust, scalable, and secure SaaS platform.

## Project Vision

To provide a comprehensive platform for developing, testing, and demonstrating adversarial attacks against various AI models, ensuring their robustness and security in real-world deployments.

## Architecture

This project adopts a microservices architecture, containerized with Docker and orchestrated by Kubernetes.

**Key Services:**
- `api_gateway`: Handles incoming requests, authentication, and routing.
- `model_service`: Manages AI model loading, inference, and versioning.
- `attack_service`: Orchestrates and executes various adversarial attacks.

## Getting Started (Local Development)

Refer to `docs/development/SETUP.md` for detailed instructions on setting up your local development environment.

## Running Tests

To run the automated tests locally:

1.  Ensure Docker Compose services are *not* running, or if they are, ensure your tests use mock services or a separate test database to prevent interference. The provided tests use in-memory SQLite databases for isolation.
2.  Navigate to the project root directory: `cd ai-adversarial-sandbox-pro`
3.  Install top-level test dependencies (pytest, httpx, etc.): `pip install pytest httpx`
4.  Install dependencies for each service locally (if you haven't already for local development):
    ```bash
    pip install -r services/api_gateway/requirements.txt
    pip install -r services/model_service/requirements.txt
    pip install -r services/attack_service/requirements.txt
    ```
    (Note: `textattack` and `torch` can be large. For quicker test runs, consider mocking their dependencies or running tests in dedicated CI environments where dependencies are pre-cached.)
5.  Run all tests: `pytest tests/`

## Licensing

This project is licensed under the [Mozilla Public License 2.0 (MPL-2.0)](LICENSE).

### What MPL-2.0 means for you

*   **Share modifications to MPL-covered files:** If you distribute versions of this project that modify files covered by MPL-2.0, you must make those modified files available under MPL-2.0. This ensures improvements to shared files remain open.
*   **Keep larger works flexible:** You may combine MPL-2.0 files with proprietary code. Only the MPL-covered files and your modifications to them must remain under MPL-2.0; surrounding modules can stay under separate terms.
*   **Preserve attribution:** Retain existing copyright notices and include the updated [NOTICE](NOTICE) file whenever you distribute the project in source or binary form.
*   **Give credit in your documentation:** When redistributing, link back to this repository and acknowledge the project authors as noted in NOTICE.

For any licensing questions or to discuss commercial collaborations, contact **lexsightllc@lexsightllc.com**.

## Documentation

- **Architecture:** `docs/architecture/README.md`
- **API Specification:** `docs/architecture/API_SPECIFICATION.md`
- **Security Policy:** `docs/security/SECURITY.md`
- **Development Setup:** `docs/development/SETUP.md`

## Current Status: Phase 1 - Foundational Development Advanced

The project now features:
- A functional microservices architecture orchestrated by Docker Compose.
- Real AI model integration (`transformers`).
- Real adversarial attack capabilities (`TextAttack`).
- Database persistence for models and attack results.
- A functional React frontend for prediction, attack launching, historical attack viewing, and model management.
- **Initial JWT-based authentication for API access.**
- **Basic automated unit and integration tests for core backend services.**
