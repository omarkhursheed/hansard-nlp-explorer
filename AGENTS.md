# Repository Guidelines

## Project Structure & Module Organization
- Source: `src/hansard/` (core packages: `crawlers/`, `parsers/`, `analysis/`, `scripts/`, `utils/`).
- Tests: top-level `tests/` (subfolders `unit/`, `integration/`) plus legacy tests in `src/hansard/tests/`.
- Data & assets: under `src/hansard/data/` (processed, metadata, content); analysis artefacts in `src/hansard/analysis/`.
- Env & tooling: `environment.yml`, `verify_all_systems.py`, repo docs in `README.md`.

## Build, Test, and Development Commands
- Create env: `conda env create -f environment.yml && conda activate hansard`.
- Quick health check: `python verify_all_systems.py` (data, imports, tests, sample analysis).
- Run tests: `pytest` or targeted: `pytest tests/unit -q`, `pytest tests/integration -q`.
- Format & lint: `black .` then `ruff .` (see coding style below).
- Run sample analysis: `python src/hansard/analysis/hansard_nlp_analysis.py --years 1920-1920 --sample 50`.
- Full pipeline (example): `bash src/hansard/scripts/run_full_processing.sh`.

## Coding Style & Naming Conventions
- Python 3.12, 4-space indent, one import per line, prefer type hints.
- Naming: modules/files `snake_case.py`; classes `PascalCase`; functions/vars `snake_case`.
- Formatting: Black (default line length 88). Linting: Ruff (fix obvious issues before PR).
- Paths: use helpers in `src/hansard/utils/path_utils.py` (avoid hard-coded absolute paths).

## Testing Guidelines
- Framework: `pytest`. Keep tests deterministic and data-light; use `tests/fixtures/` where possible.
- Structure: put new unit tests in `tests/unit/` and integration/flow tests in `tests/integration/`.
- Conventions: test files `test_*.py`; functions `test_*`. Aim to maintain/improve current ~80% coverage.
- Verify before PR: `pytest -q` and `python verify_all_systems.py` must pass.

## Commit & Pull Request Guidelines
- Commit style: Conventional Commits (examples in history: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `cleanup:`). Use a concise scope (e.g., `parsers`, `analysis`).
- PRs must include: clear description, rationale, linked issues, and any CLI examples. For analysis changes, add small before/after output snippets or path to artefacts.
- Quality gate: format + lint clean, tests passing, `verify_all_systems.py` green. Avoid large unrelated diffs.

## Security & Configuration Tips
- Do not commit large datasets or secrets. Keep configuration in code or `.env` excluded by `.gitignore`.
- Long runs: prefer scripts under `src/hansard/scripts/`; parameterize I/O paths and years. 
