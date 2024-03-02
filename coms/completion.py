from __future__ import annotations

import os
import subprocess
from pathlib import Path

import click

SHELLS = ["bash", "zsh", "fish"]
shell = os.environ.get("SHELL")

if shell is not None:
    shell = Path(shell).name
    if shell not in SHELLS:
        shell = None


@click.group(
    help="Generate shell completion",
)
def completion() -> None:
    pass


@completion.command(help="Generate shell completion code")
@click.option(
    "-s",
    "--shell",
    type=click.Choice(SHELLS),
    default=shell,
    show_default=True,
    required=shell is None,
    help="Shell type to generate for",
)
def generate(shell: str) -> None:
    subprocess.run(["MSSF"], check=True, env={**os.environ, "_MSSF_COMPLETE": f"{shell}_source"})
