# TOOLS.md

This agent uses OpenClaw's default `research` tool profile, which exposes:

- **`web_search`** — Brave Search API (via OC native; key from BRAVE_API_KEY
  in OC env). Use for finding candidate source URLs.
- **`web_fetch`** — OC native page fetch. Use for reading a specific URL
  as markdown.
- **`bash`** + **filesystem** (`read`, `write`, `edit`) — routed through
  the exec-bridge into the per-task sandbox container under `/workspace`.

The `research_fetch` skill (under `.openclaw/skills/research_fetch/`) is a
recipe — not a tool — for composing `web_fetch` with a focused LLM
extraction so each page becomes a small, citeable evidence block.

If `web_search` or `web_fetch` is unavailable in your tool list at runtime,
proceed with the tools you do have and note the limitation; do not invent
tool calls.
