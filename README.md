# Codex Session Timeline Analyzer

Read-only reports for Codex Desktop `rollout-*.jsonl` session logs.

The analyzer helps maintainers answer practical workflow questions:

- Which tool calls consumed most of a Codex review, triage, or release session?
- Did long logs, failing commands, or post-tool analysis gaps slow the turn down?
- Which sessions need manual attention before changing automation?
- Can a maintainer share an HTML report without exposing private model reasoning?

It does not call external APIs, mutate Codex session files, skip checks, or inspect private reasoning content. Reports are based on observable JSONL events, tool call spans, shell `Wall time` lines, output sizes, token counters, and inferred gaps between logged events.

## Install

```bash
python -m pip install -e .
```

## Quick Start

Analyze an anonymized sample:

```bash
codex-session-timeline examples/sample-rollout.jsonl --html-out reports/sample.html --timeline-limit -1
```

Analyze the latest local Codex Desktop session:

```bash
codex-session-timeline --latest
```

Build an HTML index for recent sessions:

```bash
codex-session-timeline --html-index-out reports/index.html --session-limit 20
```

Filter the index to one project:

```bash
codex-session-timeline --html-index-out reports/project.html --project-filter my-oss-project
```

## What It Reports

- total elapsed time and inferred non-tool time
- tool occupied union time vs cumulative logged tool time
- slowest and failed tool calls
- shell `Wall time`, exit code, timeout detection, output line/character counts
- per-turn suspected bottleneck: timeout, runtime, large output, post-tool gap, high context/reasoning, retries, or slow first action
- read-only risk labels: green, yellow, red
- HTML reports for a single session or recent-session index

## Maintainer Use Cases

- PR review: find whether test runners, `rg`, `git diff`, or large outputs dominate review time.
- Issue triage: compare several debugging sessions and identify repeated failure patterns.
- Release workflow: spot slow release checks or commands whose output is too large for efficient agent review.
- Automation design: keep a read-only baseline before changing hooks, CI gates, or agent workflows.

## Privacy And Safety

Codex private reasoning is not available in session logs and is not reported. The analyzer stores only local files that you explicitly request with `--json-out`, `--html-out`, or `--html-index-out`.

Before sharing a report publicly, review:

- commands and working directories
- tool output previews
- project names and file paths
- token counters and timestamps

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

## Status

This is an early extraction from a real maintenance workflow. The current focus is stable read-only diagnostics for maintainers before adding heavier automation.
