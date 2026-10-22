# westata

Stata-conformant econometrics with a statsmodels-style API.

Personal research tool. The design goal is that a model returns the same
numbers as the corresponding Stata command, with every claim backed by a
recorded evidence source (see `docs/superpowers/specs/`).

Status: Phase 0 (skeleton + OLS + VCE).

## Development

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[test]"
.venv\Scripts\python.exe -m pytest
```

## Documentation

- Design spec: `docs/superpowers/specs/2026-10-22-westata-design.md`
- Phase 0 plan: `docs/superpowers/plans/2026-10-22-westata-p0.md`
