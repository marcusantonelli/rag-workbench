# Long-term dev notes

Decisions and plans that span releases. Short entries; details live in the code, the
release notes and the Shammy plan.

## v1.0: rename to Shammy

Done in v1.0, not in a prerelease.

| Today | From v1.0 |
|---|---|
| RAG Workbench (desktop app) | **Shammy Workbench** |
| `.shammy` vector store and file format | **ShammyDB** |
| `rag` and `shammy` command lines | **Shammy CLI** (one `shammy` command) |
| `shammy-rag` shared retrieval library (Python, C, C#) | **Shammy Engine** |

Things the rename touches: app name, installer and bundle ids, the data folder, the CLI entry
points (keep `rag` as an alias for a release), Python package names, the `.ragworkspace`
marker and `.ragpack` extension (read the old names forever), docs, and the GitHub repository.
