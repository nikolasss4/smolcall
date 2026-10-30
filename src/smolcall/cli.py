"""Command-line entry point: ``smolcall data|train|eval|predict``."""

import json

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)
data_app = typer.Typer(no_args_is_help=True, help="Download and build the datasets.")
app.add_typer(data_app, name="data")


@app.command()
def version() -> None:
    """Print the package version."""
    from smolcall import __version__

    typer.echo(__version__)


@data_app.command("download")
def data_download() -> None:
    """Download every source at its pinned revision and verify SHA-256 hashes."""
    from smolcall.data.build import download

    download()


@data_app.command("build")
def data_build() -> None:
    """Normalize, dedup and split into data/processed/*.jsonl (fails on any leakage)."""
    from smolcall.data.build import build

    typer.echo(json.dumps(build(), indent=2))
