# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT open a public issue**
2. Email: [create a GitHub Security Advisory](https://github.com/karsten-s-nielsen/silly-kicks/security/advisories/new)
3. Include: description, reproduction steps, and potential impact

We aim to acknowledge reports within 48 hours and provide a fix timeline within 7 days.

## Security Considerations

silly-kicks is a pure computation library — it does not:
- Open network connections
- Read/write files
- Execute subprocess commands
- Deserialize untrusted data (no pickle, no yaml.load)

The primary security surface is input validation on provider DataFrames, which is enforced via `_validate_input_columns()` at every converter entry point.
