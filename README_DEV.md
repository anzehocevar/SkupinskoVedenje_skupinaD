# Setup

This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management.
Install it and run `uv sync`.
To add a dependency, run `uv add my_dependency`.
Remember to run `uv sync` every time you pull a new version from git, since someone might've added or removed a dependency.
VSCode should automatically pick up the created virtual environment.

# Code style

-   Use the suggested (`.vscode/`) formatter and type checker.
    Make sure it autoformats the code on save (should be the default, but worth mentioning in case of a VSCode moment) to avoid unnecessary diffs.
-   Use absolute imports and prefer the `from ... import ...` style. In other words, just use VSCode's auto-import.
-   Prefer strict type checking whenever possible (`# pyright: strict` at the top of a file).
    Don't make other team members decipher some spaghetti data flow (if it's hard for a type checker to understand, it's also hard for a human to understand).
    Some packages are simply not written with type checking in mind, which is why it's opt-in (and why you shouldn't enable it in notebooks), but others work fine.
    Note that "basic" type checking is always active, but won't complain about things like missing types.

## Notebooks

-   All notebooks go in `notebooks/`.
-   Notebooks are just persistent REPLs. Don't put anything too complicated inside them.
    Instead, everything should go into the `cb25d` Python module so we have a less janky interface (VSCode's notebook interface still has some annoying, but tolerable bugs), opt-in strict type checking and clean git diffs.
    Notebooks will autoreload the module whenever you make changes.
-   Inside a notebook, organize your code into sections. Use Markdown cells to label them so they show up in VSCode's outline.
    The advantage of this is that you can run each section with one click, and if you structure your code correctly, slow parts of the notebook can be skipped when they're not needed.
-   All your inputs should be inside the first cell of the notebook.
    Remember to run `Ruff: Format imports` when you change something, since VSCode won't do this automatically inside notebooks.
