# Contributing

Thanks for improving Codex Session Timeline Analyzer.

## Development Setup

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

## Contribution Scope

Good first contributions:

- parser support for additional observable Codex event shapes
- safer redaction helpers for paths, commands, and output previews
- clearer HTML report sections
- more anonymized fixtures
- tests for edge cases around missing timestamps, partial tool outputs, and long gaps

Keep the default analyzer read-only. Features that modify sessions, edit local projects, skip checks, or automatically change maintainer workflows need explicit design review.

## Pull Request Checklist

- Add or update tests for behavior changes.
- Keep sample data anonymized.
- Do not commit real Codex session logs.
- Run `pytest` and `ruff check .`.
- Update `README.md` when CLI behavior changes.
