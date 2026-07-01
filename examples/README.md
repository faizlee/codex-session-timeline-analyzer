# Examples

Generate a single-session HTML report from the anonymized sample:

```bash
codex-session-timeline examples/sample-rollout.jsonl --html-out reports/sample.html --timeline-limit -1
```

Generate an index from your local Codex Desktop sessions:

```bash
codex-session-timeline --html-index-out reports/index.html --session-limit 20
```

The analyzer reads Codex JSONL files and writes local reports only. It does not call external APIs, modify sessions, or inspect private model reasoning.
