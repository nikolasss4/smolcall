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


@app.command("train")
def train_cmd(config: str = typer.Argument(..., help="Path to configs/<name>.yaml")) -> None:
    """LoRA fine-tune with mlx_lm (Apple Silicon only)."""
    from pathlib import Path

    from smolcall.train import train

    typer.echo(f"run saved to {train(Path(config))}")


@app.command("eval")
def eval_cmd(
    system: list[str] = typer.Option(
        ..., "--system", "-s", help="e.g. base, ft-4bit, claude-haiku-5-5, replay:ft"
    ),
    split: list[str] = typer.Option(["test", "handchecked", "bfcl"], "--split"),
    limit: int | None = typer.Option(None, help="Only the first N examples (not saved as metrics)"),
) -> None:
    """Run systems on splits with the shared scorer and print headline metrics."""
    from smolcall.eval.runner import run

    for sys_name in system:
        for sp in split:
            m = run(sys_name, sp, limit)
            ci = m["accuracy"].get("ci95", [float("nan")] * 2)
            typer.echo(
                f"{m['system']:<22} {sp:<12} n={m['n']:<4} acc={m['accuracy']['value']:.3f} "
                f"[{ci[0]:.3f}, {ci[1]:.3f}]  json={m['json_valid']['value']:.3f}  "
                f"multi={(m['multi_accuracy']['value'] or 0):.3f}  "
                f"false_call={(m['false_call_rate']['value'] or 0):.3f}"
            )
