# Contributing

Thank you for your interest in improving the Adversarial AI Security Sandbox! Contributions help the project evolve and remain secure.

## License alignment

By submitting a pull request, you agree that your contribution will be licensed under the [Mozilla Public License 2.0](LICENSE). Inbound contributions are treated the same as outbound licensing ("inbound = outbound"). Please ensure every new source file includes an `SPDX-License-Identifier: MPL-2.0` header.

## Getting started

1. Fork the repository and create a feature branch.
2. Install dependencies for the services you plan to modify.
3. Run the relevant unit tests with `pytest tests/` and `yarn test` for the frontend as needed.
4. Run `pre-commit install` to enable automated SPDX header insertion and formatting checks.
5. Submit a pull request with a clear description of your changes and reference any related issues.

For security-sensitive disclosures, please follow the guidance in [`docs/security/SECURITY.md`](docs/security/SECURITY.md).
