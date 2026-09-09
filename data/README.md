# Data

Nothing in `data/raw/` or `data/processed/` is tracked by git — only this file is.

| Directory        | Contents                                              |
| ---------------- | ----------------------------------------------------- |
| `data/raw/`      | Downloaded or received data, never edited in place.    |
| `data/processed/`| Everything derived from `raw/` by code in this repo.   |

`data/processed/` should always be reproducible from `data/raw/`:

```bash
uv run scripts/download_data.py      # populates data/raw/
```

Record the source, licence and download date of each dataset here as it is added.
