"""
cli.py — PrivaRepo Command-Line Interface

Production-grade CLI built with Typer + Rich.
Every command has progress bars, syntax highlighting, and structured output.

Commands:
  index       — Index a code repository
  query       — Ask a question about the codebase (full RAG)
  search      — Raw hybrid search (no LLM)
  stats       — Show collection statistics
  benchmark   — Run latency / throughput benchmark
  evaluate    — Run full evaluation suite
  interactive — Start interactive chat session
  export      — Export collection to NDJSON
  import      — Import collection from NDJSON
  reset       — Reset (drop) the collection
  serve       — Start the Flask REST API server
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from config import AppConfig
from repository_manager import RepositoryManager

app = typer.Typer(
    name="privarepo",
    help="🔒 PrivaRepo — Fully Local AI Code Intelligence Assistant",
    add_completion=True,
    rich_markup_mode="rich",
    no_args_is_help=True,
)
console = Console()

# ---------------------------------------------------------------------------
# Repository management — Phase 1
#
# Bookkeeping only: registers {name, path, collection} entries and tracks
# which one is "active" via repository_manager.RepositoryManager. Deliberately
# does not touch rag_pipeline/vector_store/evaluator, and no other command
# in this file consumes the registry yet — that's a later phase.
# ---------------------------------------------------------------------------

repo_app = typer.Typer(
    name="repo",
    help="📁 Manage multiple indexed repositories.",
    no_args_is_help=True,
)
app.add_typer(repo_app, name="repo")

_repo_manager = RepositoryManager()


@repo_app.command("add")
def cmd_repo_add(
    path: str = typer.Argument(..., help="Path to the repository to register"),
    name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Display name (defaults to the folder name)"
    ),
) -> None:
    """➕ Register a repository for multi-repo indexing/querying."""
    repo_path = Path(path).resolve()
    if not repo_path.is_dir():
        console.print(f"[red]Error:[/red] '{repo_path}' is not a valid directory.")
        raise typer.Exit(1)

    display_name = name or repo_path.name

    try:
        entry = _repo_manager.add(str(repo_path), name=display_name)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    console.print()
    console.print(
        Panel(
            f"[bold]Name[/bold]       : {entry['name']}\n"
            f"[bold]Path[/bold]       : {entry['path']}\n"
            f"[bold]Collection[/bold] : {entry['collection']}",
            title="[bold green]✅ Repository Added[/bold green]",
            border_style="green",
        )
    )
    console.print(f"[dim]Run [bold]privarepo repo select {entry['name']}[/bold] to make it active.[/dim]")


@repo_app.command("list")
def cmd_repo_list() -> None:
    """📋 List all registered repositories."""
    repos = _repo_manager.list()
    if not repos:
        console.print(
            "[dim]No repositories registered yet. "
            "Run [bold]privarepo repo add <path>[/bold].[/dim]"
        )
        return

    active = _repo_manager.get_active()
    active_name = active["name"] if active else None

    table = Table(title="Repositories", box=box.ROUNDED, border_style="cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("Name", style="bold white")
    table.add_column("Path", style="yellow")
    table.add_column("Collection", style="cyan")
    table.add_column("Active", style="green", width=8)

    for i, r in enumerate(repos, start=1):
        mark = "●" if r["name"] == active_name else ""
        table.add_row(str(i), r["name"], r["path"], r["collection"], mark)

    console.print(table)


@repo_app.command("select")
def cmd_repo_select(
    name: str = typer.Argument(..., help="Name of a registered repository"),
) -> None:
    """🎯 Set the active repository."""
    try:
        entry = _repo_manager.select(name)
    except KeyError:
        console.print(
            f"[red]Error:[/red] No registered repository named '{name}'. "
            "Run [bold]privarepo repo list[/bold]."
        )
        raise typer.Exit(1)

    console.print(f"\n[green]✅ Active repository changed to[/green] [bold]{entry['name']}[/bold]")


@repo_app.command("remove")
def cmd_repo_remove(
    name: str = typer.Argument(..., help="Name of a registered repository to remove"),
) -> None:
    """🗑  Unregister a repository. This only removes the registry entry —
    it does not delete any indexed data (no vector store / BM25 changes)."""
    if not Confirm.ask(f"[bold red]⚠  Remove '{name}' from the repository registry?[/bold red]"):
        console.print("[dim]Aborted.[/dim]")
        raise typer.Exit(0)

    try:
        _repo_manager.remove(name)
    except KeyError:
        console.print(f"[red]Error:[/red] No registered repository named '{name}'.")
        raise typer.Exit(1)

    console.print("\n[green]Repository unregistered.[/green]")


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_config(config_overrides: Optional[dict] = None) -> AppConfig:
    cfg = AppConfig()
    if config_overrides:
        for key, val in config_overrides.items():
            if hasattr(cfg, key):
                setattr(cfg, key, val)
    cfg.validate()
    return cfg


# ---------------------------------------------------------------------------
# Rich helpers
# ---------------------------------------------------------------------------

def _print_header() -> None:
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]PrivaRepo[/bold cyan] [dim]— Fully Local AI Code Intelligence[/dim]",
            border_style="cyan",
        )
    )
    console.print()


def _print_chunk_table(chunks: list, title: str = "Search Results") -> None:
    table = Table(
        title=title,
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
        expand=True,
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Type", style="cyan", width=10)
    table.add_column("Name", style="bold white", max_width=30)
    table.add_column("File", style="yellow", max_width=40)
    table.add_column("Lines", style="green", width=10)
    table.add_column("Lang", style="blue", width=8)
    table.add_column("Score", style="magenta", width=8)

    for i, chunk in enumerate(chunks, start=1):
        name = chunk.function_name or chunk.class_name or "—"
        file_display = str(Path(chunk.file_path).name) if chunk.file_path else "—"
        lines = f"{chunk.start_line}–{chunk.end_line}"
        score = f"{chunk.rerank_score:.3f}" if chunk.rerank_score else "—"
        table.add_row(
            str(i), chunk.chunk_type, name, file_display, lines, chunk.language, score
        )

    console.print(table)


def _print_rag_response(response, show_reasoning: bool = True) -> None:
    """Render a RAGResponse with Rich formatting."""
    # Answer panel
    console.print(
        Panel(
            Markdown(response.answer),
            title="[bold green]Answer[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )

    # Reasoning (collapsible)
    if show_reasoning and response.reasoning:
        console.print(
            Panel(
                Markdown(response.reasoning),
                title="[dim]Reasoning[/dim]",
                border_style="dim",
            )
        )

    # Citation table
    if response.retrieved_chunks:
        console.print()
        console.print("[bold]Sources used:[/bold]")
        cite_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        cite_table.add_column("Rank", style="dim", width=5)
        cite_table.add_column("Type", style="cyan", width=10)
        cite_table.add_column("Name", style="bold white", max_width=30)
        cite_table.add_column("File", style="yellow", max_width=45)
        cite_table.add_column("L.", style="green", width=12)
        cite_table.add_column("Rerank", style="magenta", width=8)

        for chunk in response.retrieved_chunks:
            name = chunk.function_name or chunk.class_name or "—"
            file_short = str(Path(chunk.file_path).name) if chunk.file_path else "—"
            lines = f"{chunk.start_line}–{chunk.end_line}"
            score = f"{chunk.rerank_score:.3f}" if chunk.rerank_score else "—"
            cite_table.add_row(
                str(chunk.final_rank), chunk.chunk_type, name, file_short, lines, score
            )
        console.print(cite_table)

    # Timing footer
    console.print(
        f"[dim]⏱  Retrieval: {response.retrieval_time*1000:.0f}ms  "
        f"| Generation: {response.generation_time*1000:.0f}ms  "
        f"| Total: {response.total_time*1000:.0f}ms  "
        f"| Model: {response.model}[/dim]"
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command("index")
def cmd_index(
    repo_path: Optional[str] = typer.Argument(
        None,
        help="Path to the repository to index. If omitted, uses the active "
             "repository set via `privarepo repo select`."
    ),
    language: Optional[str] = typer.Option(
        None, "--language", "-l",
        help="Only index files of this language (python/java/javascript/typescript)"
    ),
    reset: bool = typer.Option(
        False, "--reset", "-r",
        help="Reset the collection before indexing"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """📦 Index a code repository into the vector store and BM25 index."""
    _setup_logging(verbose)
    _print_header()

    if repo_path:
        repo = Path(repo_path).resolve()
    else:
        active = _repo_manager.get_active()
        if active is None:
            console.print(
                "[red]Error:[/red] No repository path given and no active repository set.\n"
                "Either provide a path — [bold]privarepo index <path>[/bold] — "
                "or run [bold]privarepo repo select <name>[/bold] first."
            )
            raise typer.Exit(1)
        repo = Path(active["path"]).resolve()
        console.print(f"[dim]Using active repository:[/dim] [bold]{active['name']}[/bold]")

    if not repo.is_dir():
        console.print(f"[red]Error:[/red] '{repo}' is not a valid directory.")
        raise typer.Exit(1)

    console.print(f"[bold]Indexing:[/bold] {repo}")
    if language:
        console.print(f"[dim]Language filter: {language}[/dim]")
    if reset:
        console.print("[yellow]⚠  Reset mode — existing data will be cleared.[/yellow]")
        if not Confirm.ask("Are you sure you want to reset the collection?"):
            raise typer.Exit(0)

    cfg = _load_config()

    from rag_pipeline import RAGPipeline
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Initialising pipeline...", total=None)
        pipeline = RAGPipeline(cfg)
        if reset:
            pipeline.reset()
        progress.update(task, description="Parsing & indexing repository...")
        result = pipeline.index_repository(
            repo_path=str(repo),
            include_extensions=[language] if language else None,
            show_progress=False,
        )
        progress.update(task, description="Done!", completed=True)

    if result["status"] == "success":
        # Summary table
        table = Table(box=box.ROUNDED, border_style="green")
        table.add_column("Metric", style="bold")
        table.add_column("Value", style="green")
        table.add_row("Chunks extracted", str(result["chunks_extracted"]))
        table.add_row("Chunks indexed", str(result["chunks_indexed"]))
        table.add_row("Collection total", str(result["collection_total"]))
        table.add_row("Parse time", f"{result['parse_time_seconds']:.2f}s")
        table.add_row("Index time", f"{result['index_time_seconds']:.2f}s")
        table.add_row("BM25 build time", f"{result['bm25_build_time_seconds']:.2f}s")
        table.add_row("Total time", f"{result['total_time_seconds']:.2f}s")
        console.print()
        console.print(Panel(table, title="[bold green]✅ Indexing Complete[/bold green]", border_style="green"))
    else:
        console.print(f"[yellow]⚠  {result.get('message', 'Indexing incomplete.')}[/yellow]")


@app.command("query")
def cmd_query(
    question: str = typer.Argument(..., help="Your question about the codebase"),
    task: str = typer.Option(
        "general", "--task", "-t",
        help="Task type: general | explain | find_bugs | similar_code | function_search | class_search"
    ),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Filter by language"),
    chunk_type: Optional[str] = typer.Option(None, "--type", help="Filter by chunk type"),
    file: Optional[str] = typer.Option(None, "--file", "-f", help="Filter by file path"),
    no_reasoning: bool = typer.Option(False, "--no-reasoning", help="Hide the reasoning section"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """🧠 Ask a question about the indexed codebase (full RAG pipeline)."""
    _setup_logging(verbose)
    _print_header()

    cfg = _load_config()

    from rag_pipeline import RAGPipeline
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        prog_task = progress.add_task("Retrieving relevant code...", total=None)
        pipeline = RAGPipeline(cfg)
        progress.update(prog_task, description="Generating answer with Ollama...")

        if not pipeline.llm.is_available():
            console.print(
                "[red]Error:[/red] Ollama is not available. "
                "Start Ollama and ensure the model is pulled:\n"
                f"  [bold]ollama pull {cfg.llm.model}[/bold]"
            )
            raise typer.Exit(1)

        response = pipeline.query(
            question=question,
            task_type=task,
            language=language,
            chunk_type=chunk_type,
            file_path=file,
        )
        progress.update(prog_task, description="Done!", completed=True)

    console.print()
    console.rule(f"[bold cyan]Q: {question[:80]}{'...' if len(question)>80 else ''}[/bold cyan]")
    console.print()
    _print_rag_response(response, show_reasoning=not no_reasoning)


@app.command("search")
def cmd_search(
    query: str = typer.Argument(..., help="Search query"),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Filter by language"),
    chunk_type: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by chunk type"),
    show_code: bool = typer.Option(False, "--code", "-c", help="Show code snippets inline"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """🔍 Raw hybrid search (vector + BM25, no LLM generation)."""
    _setup_logging(verbose)
    _print_header()

    cfg = _load_config()

    from rag_pipeline import RAGPipeline
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("Searching...", total=None)
        pipeline = RAGPipeline(cfg)
        chunks, timings = pipeline.search(query=query, language=language, chunk_type=chunk_type)
        progress.update(task, description="Done!")

    console.print()
    console.rule(f"[bold cyan]Search: {query}[/bold cyan]")
    console.print()

    if not chunks:
        console.print("[yellow]No results found.[/yellow]")
        return

    _print_chunk_table(chunks, title=f"Top {len(chunks)} Results")

    if show_code:
        console.print()
        for chunk in chunks:
            lang = chunk.language or "text"
            syntax = Syntax(
                chunk.raw_code[:1200],
                lang,
                line_numbers=True,
                start_line=chunk.start_line,
                theme="monokai",
                word_wrap=True,
            )
            name = chunk.function_name or chunk.class_name or "code"
            console.print(
                Panel(
                    syntax,
                    title=f"[bold]{name}[/bold] — {Path(chunk.file_path).name}",
                    border_style="cyan",
                )
            )

    console.print(
        f"\n[dim]⏱  Total: {timings.get('total_ms', 0):.1f}ms  "
        f"(Vector: {timings.get('vector_ms', 0):.1f}ms, "
        f"BM25: {timings.get('bm25_ms', 0):.1f}ms, "
        f"RRF: {timings.get('rrf_ms', 0):.1f}ms, "
        f"Rerank: {timings.get('rerank_ms', 0):.1f}ms)[/dim]"
    )


@app.command("stats")
def cmd_stats(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """📊 Show collection statistics."""
    _setup_logging(verbose)
    _print_header()

    cfg = _load_config()

    from rag_pipeline import RAGPipeline
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task("Loading stats...", total=None)
        pipeline = RAGPipeline(cfg)
        stats = pipeline.get_stats()
        progress.update(task, completed=True)

    console.print()

    # Main stats table
    main_table = Table(title="Collection Statistics", box=box.ROUNDED, border_style="cyan")
    main_table.add_column("Metric", style="bold")
    main_table.add_column("Value", style="cyan")
    main_table.add_row("Total Chunks", str(stats.get("total_chunks", 0)))
    main_table.add_row("Unique Files", str(stats.get("unique_files", 0)))
    main_table.add_row("Collection Name", stats.get("collection_name", "—"))
    main_table.add_row("Embedding Model", stats.get("embedding_model", "—"))
    main_table.add_row("Embedding Dim", str(stats.get("embedding_dimension", "—")))
    main_table.add_row("Persist Dir", stats.get("persist_dir", "—"))
    main_table.add_row("BM25 Index Size", str(stats.get("bm25_index_size", 0)))
    console.print(main_table)

    # Language breakdown
    if stats.get("languages"):
        console.print()
        lang_table = Table(title="Language Breakdown", box=box.SIMPLE, border_style="dim")
        lang_table.add_column("Language", style="bold cyan")
        lang_table.add_column("Chunks", style="green")
        for lang, count in sorted(stats["languages"].items(), key=lambda x: x[1], reverse=True):
            lang_table.add_row(lang, str(count))
        console.print(lang_table)

    # Chunk type breakdown
    if stats.get("chunk_types"):
        console.print()
        type_table = Table(title="Chunk Type Breakdown", box=box.SIMPLE, border_style="dim")
        type_table.add_column("Type", style="bold magenta")
        type_table.add_column("Chunks", style="green")
        for ct, count in sorted(stats["chunk_types"].items(), key=lambda x: x[1], reverse=True):
            type_table.add_row(ct, str(count))
        console.print(type_table)


@app.command("benchmark")
def cmd_benchmark(
    runs: int = typer.Option(20, "--runs", "-n", help="Number of benchmark runs"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output JSON path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """⚡ Run latency benchmark and save results."""
    _setup_logging(verbose)
    _print_header()

    cfg = _load_config()

    from evaluator import Evaluator
    from rag_pipeline import RAGPipeline

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task(f"Running {runs} benchmark iterations...", total=None)
        pipeline = RAGPipeline(cfg)
        evaluator = Evaluator(pipeline, cfg.eval)
        result = evaluator.run_latency_benchmark(n_runs=runs)
        progress.update(task, description="Done!")

    console.print()
    table = Table(title="⚡ Latency Benchmark Results", box=box.ROUNDED, border_style="yellow")
    table.add_column("Metric", style="bold")
    table.add_column("P50 (ms)", style="green")
    table.add_column("P95 (ms)", style="yellow")
    table.add_column("P99 (ms)", style="red")
    table.add_row("Retrieval", f"{result.retrieval_p50_ms:.1f}", f"{result.retrieval_p95_ms:.1f}", f"{result.retrieval_p99_ms:.1f}")
    table.add_row("Generation", f"{result.generation_p50_ms:.1f}", f"{result.generation_p95_ms:.1f}", f"{result.generation_p99_ms:.1f}")
    table.add_row("Total", f"{result.total_p50_ms:.1f}", f"{result.total_p95_ms:.1f}", f"{result.total_p99_ms:.1f}")
    console.print(table)

    console.print(f"\n[dim]Runs: {result.n_runs} | Memory Δ: {result.memory_usage_mb:.1f} MB | Collection: {result.collection_size} chunks[/dim]")

    # Save
    report = __import__("evaluator").FullBenchmarkReport(
        timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        system_info={},
        collection_stats=pipeline.get_stats(),
        latency=result,
    )
    path = evaluator.save_report(report, output)
    console.print(f"\n[green]✅ Report saved:[/green] {path}")


@app.command("evaluate")
def cmd_evaluate(
    queries_path: Optional[str] = typer.Option(
        None, "--queries", "-q", help="Path to eval_queries.json"
    ),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output JSON path"),
    skip_generation: bool = typer.Option(False, "--skip-generation", help="Skip generation evaluation"),
    skip_latency: bool = typer.Option(False, "--skip-latency", help="Skip latency benchmark"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """🧪 Run the full evaluation suite (retrieval + generation + latency)."""
    _setup_logging(verbose)
    _print_header()

    cfg = _load_config()

    from evaluator import Evaluator
    from rag_pipeline import RAGPipeline

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("Running full evaluation...", total=None)
        pipeline = RAGPipeline(cfg)
        evaluator = Evaluator(pipeline, cfg.eval)
        queries = evaluator.load_eval_queries(queries_path)
        report = evaluator.run_full_benchmark(
            queries=queries,
            include_retrieval=True,
            include_generation=not skip_generation,
            include_latency=not skip_latency,
        )
        progress.update(task, description="Done!")

    console.print()

    if report.retrieval:
        r = report.retrieval
        ret_table = Table(title="📊 Retrieval Metrics", box=box.ROUNDED, border_style="green")
        ret_table.add_column("Metric", style="bold")
        ret_table.add_column("Score", style="green")
        ret_table.add_column("Target", style="dim")
        ret_table.add_row(f"Precision@{r.k}", f"{r.precision_at_k:.3f}", "≥ 0.70")
        ret_table.add_row(f"Recall@{r.k}", f"{r.recall_at_k:.3f}", "≥ 0.95")
        ret_table.add_row("MRR", f"{r.mrr:.3f}", "—")
        ret_table.add_row("Hit Rate", f"{r.hit_rate:.3f}", "—")
        console.print(ret_table)

    if report.generation:
        g = report.generation
        gen_table = Table(title="🤖 Generation Metrics", box=box.ROUNDED, border_style="cyan")
        gen_table.add_column("Metric", style="bold")
        gen_table.add_column("Score", style="cyan")
        gen_table.add_column("Target", style="dim")
        gen_table.add_row("Faithfulness", f"{g.faithfulness:.3f}", "≥ 0.90")
        gen_table.add_row("Answer Relevancy", f"{g.answer_relevancy:.3f}", "≥ 0.90")
        gen_table.add_row("Context Precision", f"{g.context_precision:.3f}", "—")
        gen_table.add_row("Context Recall", f"{g.context_recall:.3f}", "—")
        console.print()
        console.print(gen_table)

    path = evaluator.save_report(report, output)
    console.print(f"\n[green]✅ Full report saved:[/green] {path}")


@app.command("interactive")
def cmd_interactive(
    language: Optional[str] = typer.Option(None, "--language", "-l"),
    task: str = typer.Option("general", "--task", "-t"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """💬 Start an interactive chat session with the codebase."""
    _setup_logging(verbose)
    _print_header()

    cfg = _load_config()

    from rag_pipeline import RAGPipeline

    pipeline = RAGPipeline(cfg)

    if not pipeline.llm.is_available():
        console.print(
            f"[red]Error:[/red] Ollama not available. Run: [bold]ollama pull {cfg.llm.model}[/bold]"
        )
        raise typer.Exit(1)

    console.print(
        Panel(
            "[bold green]PrivaRepo Interactive Mode[/bold green]\n"
            "[dim]Ask questions about your codebase. Type 'exit' or 'quit' to leave.\n"
            "Commands: /search <query>, /stats, /reset-history[/dim]",
            border_style="green",
        )
    )
    console.print()

    conversation_history: list = []
    turn = 0

    while True:
        try:
            question = Prompt.ask(f"[bold cyan]You[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "bye"):
            console.print("[dim]Goodbye![/dim]")
            break
        if question.lower() == "/stats":
            stats = pipeline.get_stats()
            console.print_json(json.dumps(stats, indent=2))
            continue
        if question.lower() == "/reset-history":
            conversation_history = []
            console.print("[dim]Conversation history cleared.[/dim]")
            continue
        if question.lower().startswith("/search "):
            raw_q = question[8:].strip()
            chunks, _ = pipeline.search(raw_q, language=language)
            _print_chunk_table(chunks, title=f"Search: {raw_q}")
            continue

        turn += 1
        with console.status("[bold green]Thinking...[/bold green]", spinner="dots"):
            try:
                response = pipeline.query(
                    question=question,
                    task_type=task,
                    language=language,
                )
                # Maintain conversation context
                conversation_history.append({"role": "user", "content": question})
                conversation_history.append({"role": "assistant", "content": response.answer})
                # Keep last 10 turns
                if len(conversation_history) > 20:
                    conversation_history = conversation_history[-20:]
            except Exception as exc:
                console.print(f"[red]Error:[/red] {exc}")
                continue

        console.print()
        console.print(
            Panel(
                Markdown(response.answer),
                title=f"[bold green]PrivaRepo[/bold green] [dim](turn {turn})[/dim]",
                border_style="green",
                padding=(1, 2),
            )
        )
        console.print(
            f"[dim]⏱  {response.retrieval_time*1000:.0f}ms retrieval · "
            f"{response.generation_time*1000:.0f}ms generation · "
            f"{len(response.retrieved_chunks)} chunks[/dim]"
        )
        console.print()


@app.command("export")
def cmd_export(
    output: str = typer.Argument(..., help="Output path for the NDJSON export"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """📤 Export the vector store collection to a NDJSON file."""
    _setup_logging(verbose)
    _print_header()

    cfg = _load_config()

    from rag_pipeline import RAGPipeline
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("Exporting collection...", total=None)
        pipeline = RAGPipeline(cfg)
        n = pipeline.export(output)
        progress.update(task, description="Done!")

    console.print(f"\n[green]✅ Exported {n} records to:[/green] {output}")


@app.command("import")
def cmd_import(
    input_path: str = typer.Argument(..., help="Path to the NDJSON export file"),
    reset: bool = typer.Option(False, "--reset", "-r", help="Reset collection before import"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """📥 Import a collection from a previously exported NDJSON file."""
    _setup_logging(verbose)
    _print_header()

    if reset and not Confirm.ask("Reset collection before import?"):
        raise typer.Exit(0)

    cfg = _load_config()

    from rag_pipeline import RAGPipeline
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("Importing collection...", total=None)
        pipeline = RAGPipeline(cfg)
        n = pipeline.import_data(input_path, reset_first=reset)
        progress.update(task, description="Done!")

    console.print(f"\n[green]✅ Imported {n} records from:[/green] {input_path}")


@app.command("reset")
def cmd_reset(
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """🗑  Reset the vector store and BM25 index (irreversible)."""
    _setup_logging(verbose)
    _print_header()

    if not force and not Confirm.ask(
        "[bold red]⚠  This will permanently delete all indexed data. Proceed?[/bold red]"
    ):
        console.print("[dim]Aborted.[/dim]")
        raise typer.Exit(0)

    cfg = _load_config()

    from rag_pipeline import RAGPipeline
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task("Resetting...", total=None)
        pipeline = RAGPipeline(cfg)
        pipeline.reset()
        progress.update(task, description="Done!")

    console.print("\n[green]✅ Collection reset successfully.[/green]")


@app.command("serve")
def cmd_serve(
    host: str = typer.Option("127.0.0.1", "--host", "-H"),
    port: int = typer.Option(8080, "--port", "-p"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """🌐 Start the Flask REST API server (optional UI mode)."""
    _setup_logging(verbose)
    _print_header()

    try:
        from flask import Flask, jsonify, request
        from flask_cors import CORS
    except ImportError:
        console.print("[red]Flask not installed. Run: pip install flask flask-cors[/red]")
        raise typer.Exit(1)

    cfg = _load_config()

    from rag_pipeline import RAGPipeline
    console.print("Initialising pipeline...")
    pipeline = RAGPipeline(cfg)

    flask_app = Flask("privarepo")
    CORS(flask_app)

    @flask_app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "collection_chunks": pipeline.vector_store.count})

    @flask_app.route("/stats", methods=["GET"])
    def stats():
        return jsonify(pipeline.get_stats())

    @flask_app.route("/query", methods=["POST"])
    def query():
        data = request.json or {}
        question = data.get("question", "")
        if not question:
            return jsonify({"error": "question is required"}), 400
        response = pipeline.query(
            question=question,
            task_type=data.get("task_type", "general"),
            language=data.get("language"),
            chunk_type=data.get("chunk_type"),
        )
        return jsonify({
            "answer": response.answer,
            "reasoning": response.reasoning,
            "referenced_files": response.referenced_files,
            "functions_used": response.functions_used,
            "retrieval_time_ms": response.retrieval_time * 1000,
            "generation_time_ms": response.generation_time * 1000,
            "total_time_ms": response.total_time * 1000,
            "model": response.model,
        })

    @flask_app.route("/search", methods=["POST"])
    def search():
        data = request.json or {}
        query_text = data.get("query", "")
        if not query_text:
            return jsonify({"error": "query is required"}), 400
        chunks, timings = pipeline.search(
            query=query_text,
            language=data.get("language"),
            chunk_type=data.get("chunk_type"),
        )
        return jsonify({
            "results": [
                {
                    "chunk_id": c.chunk_id,
                    "chunk_type": c.chunk_type,
                    "function_name": c.function_name,
                    "class_name": c.class_name,
                    "file_path": c.file_path,
                    "language": c.language,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "rerank_score": c.rerank_score,
                    "code": c.raw_code,
                }
                for c in chunks
            ],
            "timings_ms": timings,
        })

    console.print(
        Panel(
            f"[bold green]🌐 PrivaRepo API Server[/bold green]\n"
            f"[dim]Listening on http://{host}:{port}\n"
            f"Endpoints: GET /health, GET /stats, POST /query, POST /search[/dim]",
            border_style="green",
        )
    )
    flask_app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    app()