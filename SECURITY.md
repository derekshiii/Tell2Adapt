# Security Policy

## Reporting a vulnerability

Please do not report security vulnerabilities through public GitHub issues.
Contact the corresponding author privately using the contact information in
the Tell2Adapt paper.

Include the following where possible:

- affected version or commit;
- affected file and line;
- description and impact;
- reproduction steps or a minimal proof of concept;
- whether patient data, credentials, or model weights are involved.

Issues in upstream dependencies should be reported to the upstream project.
Tell2Adapt follows coordinated disclosure for vulnerabilities in this code.

## Scope

This policy covers the Tell2Adapt source code released in this repository.
Datasets, model checkpoints, and third-party assets remain subject to their
original licenses and security policies.

## Data and credential handling

- Never commit patient data, case identifiers, model weights, or experiment
  outputs.
- Never commit API keys, tokens, passwords, private keys, or `.env` files.
- Load pickled NPZ files only from trusted sources; pickle can execute code.
- Use private reporting for any suspected exposure of sensitive medical data.

## Supported versions

Security fixes are applied to the latest release on the default branch.
