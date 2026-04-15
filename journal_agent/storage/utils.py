from pathlib import Path

def resolve_project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate

    return Path.cwd().resolve()