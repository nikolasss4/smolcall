"""Command-line entry point. Subcommands are added as the pipeline grows."""

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def version() -> None:
    """Print the package version."""
    from smolcall import __version__

    typer.echo(__version__)
