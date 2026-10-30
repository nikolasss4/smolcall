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


@app.command("bench")
def bench_cmd(
    system: list[str] = typer.Option(..., "--system", "-s"),
    n: int = typer.Option(40, help="Measured requests (after 3 warmup requests)"),
) -> None:
    """Single-request latency, tokens/sec, peak memory and disk size of local models."""
    from smolcall.eval.bench import bench

    for s in system:
        typer.echo(json.dumps(bench(s, n), indent=2))


@app.command("predict")
def predict_cmd(
    query: str = typer.Argument(..., help="The user request"),
    tools: str = typer.Option("examples/tools.json", help="JSON file with a list of tool schemas"),
    model: str = typer.Option("models/smolcall-4bit", help="Model directory or HF repo"),
    adapter: str | None = typer.Option(None, help="Optional LoRA adapter directory"),
    raw: bool = typer.Option(False, help="Also print the raw model output"),
) -> None:
    """Predict the tool calls for one request."""
    from pathlib import Path

    from smolcall.predict import load_tools, predict

    calls, text = predict(query, load_tools(Path(tools)), model, adapter)
    if raw:
        typer.echo(f"raw: {text}")
    typer.echo(json.dumps(calls, indent=2, ensure_ascii=False))


@app.command("prompt")
def prompt_cmd(
    tools: str = typer.Option("examples/tools.json", help="JSON file with a list of tool schemas"),
) -> None:
    """Print the system prompt for these tools (for use with mlx_lm.server or any client)."""
    from pathlib import Path

    from smolcall.predict import load_tools
    from smolcall.prompt import system_prompt

    typer.echo(system_prompt(load_tools(Path(tools))))
