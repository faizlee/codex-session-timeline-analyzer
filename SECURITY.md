# Security And Privacy

This tool analyzes local Codex Desktop JSONL session logs. Those logs can contain command text, working directories, output previews, timestamps, and project names.

## Reporting Issues

Please open a GitHub security advisory or contact the maintainer privately for vulnerabilities that could expose sensitive local data.

## Sharing Reports Safely

Before publishing generated JSON or HTML reports, manually review:

- command arguments
- working directories
- tool output previews
- project names
- timestamps
- token counters

The analyzer does not access private model reasoning. It also does not call external APIs or upload reports.

## Safe Defaults

- Read-only input handling.
- Local output only.
- No network calls.
- No automatic workflow changes.
- No session mutation.
