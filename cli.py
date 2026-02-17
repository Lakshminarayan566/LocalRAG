#!/usr/bin/env python3
"""
PrivaRepo CLI
Command-line interface for local code intelligence
"""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console

from config import SystemConfig, load_config, save_config
from rag_pipeline import PrivaRepoRAG
from evaluator import RAGASEvaluator


app = typer.Typer(
    name="privarepo",
    help="Local RAG for Code Intelligence - Secure, Private, Fast",
    add_completion=False
)
console = Console()


@app.command()
def index(
    path: str = typer.Argument(..., help="Path to codebase to index"),
    extensions: Optional[str] = typer.Option(
        None,
        "--ext",
        help="File extensions to index (comma-separated, e.g., '.py,.js,.java')"
    ),
    config_path: Optional[str] = typer.Option(None, "--config", help="Path to config file")
):
    """Index a codebase for semantic search"""
    
    config = load_config(config_path)
    rag = PrivaRepoRAG(config)
    
    # Parse extensions
    file_extensions = None
    if extensions:
        file_extensions = [ext.strip() for ext in extensions.split(',')]
    
    # Index codebase
    stats = rag.index_codebase(path, file_extensions=file_extensions)
    
    console.print(f"\n[bold green]✓[/bold green] Successfully indexed {stats['files_processed']} files")


@app.command()
def query(
    question: str = typer.Argument(..., help="Question about the codebase"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to retrieve"),
    language: Optional[str] = typer.Option(None, "--lang", "-l", help="Filter by programming language"),
    config_path: Optional[str] = typer.Option(None, "--config", help="Path to config file")
):
    """Query the indexed codebase"""
    
    config = load_config(config_path)
    rag = PrivaRepoRAG(config)
    
    result = rag.query(
        question=question,
        top_k=top_k,
        language_filter=language
    )
    
    # Results are already printed by the query method
    pass


@app.command()
def interactive(
    config_path: Optional[str] = typer.Option(None, "--config", help="Path to config file")
):
    """Start interactive query session"""
    
    config = load_config(config_path)
    rag = PrivaRepoRAG(config)
    
    rag.interactive_mode()


@app.command()
def explain(
    file_path: str = typer.Argument(..., help="Path to file to explain"),
    function: Optional[str] = typer.Option(None, "--function", "-f", help="Specific function to explain"),
    config_path: Optional[str] = typer.Option(None, "--config", help="Path to config file")
):
    """Explain code from a file or function"""
    
    config = load_config(config_path)
    rag = PrivaRepoRAG(config)
    
    explanation = rag.explain_code(file_path, function)
    
    console.print(f"\n[bold]Explanation:[/bold]")
    console.print(explanation)


@app.command()
def search(
    description: str = typer.Argument(..., help="Describe the functionality to search for"),
    chunk_type: Optional[str] = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter by chunk type (function, class, method)"
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
    config_path: Optional[str] = typer.Option(None, "--config", help="Path to config file")
):
    """Search for code by functionality description"""
    
    config = load_config(config_path)
    rag = PrivaRepoRAG(config)
    
    results = rag.search_by_functionality(description, chunk_type, top_k)
    
    console.print(f"\n[bold]Found {len(results)} results:[/bold]\n")
    
    for i, result in enumerate(results, 1):
        metadata = result['metadata']
        console.print(f"[cyan]{i}. {metadata['file_path']}:{metadata['start_line']}[/cyan]")
        console.print(f"   Type: {metadata['chunk_type']} | Similarity: {result['similarity']:.3f}")
        if metadata.get('name'):
            console.print(f"   Name: {metadata['name']}")
        console.print()


@app.command()
def stats(
    config_path: Optional[str] = typer.Option(None, "--config", help="Path to config file")
):
    """Show codebase statistics"""
    
    config = load_config(config_path)
    rag = PrivaRepoRAG(config)
    
    rag.get_codebase_summary()


@app.command()
def benchmark(
    config_path: Optional[str] = typer.Option(None, "--config", help="Path to config file"),
    save_results: Optional[str] = typer.Option(None, "--save", help="Save results to file")
):
    """Run system performance benchmark"""
    
    config = load_config(config_path)
    rag = PrivaRepoRAG(config)
    
    results = rag.benchmark_system()
    
    if save_results:
        import json
        with open(save_results, 'w') as f:
            json.dump(results, f, indent=2)
        console.print(f"\n[bold green]✓[/bold green] Results saved to {save_results}")


@app.command()
def evaluate(
    config_path: Optional[str] = typer.Option(None, "--config", help="Path to config file"),
    test_suite: Optional[str] = typer.Option(None, "--test-suite", help="Path to custom test suite JSON"),
    save_results: Optional[str] = typer.Option(None, "--save", help="Save results to file")
):
    """Run RAGAS evaluation"""
    
    config = load_config(config_path)
    rag = PrivaRepoRAG(config)
    evaluator = RAGASEvaluator(rag)
    
    if test_suite:
        import json
        with open(test_suite, 'r') as f:
            test_cases = json.load(f)
    else:
        test_cases = evaluator.create_test_suite()
    
    results = evaluator.run_comprehensive_evaluation()
    
    if save_results:
        evaluator.export_results(results, save_results)


@app.command()
def config_init(
    output: str = typer.Option("privarepo_config.json", "--output", "-o", help="Output config file path")
):
    """Generate default configuration file"""
    
    config = SystemConfig()
    save_config(config, output)
    
    console.print(f"[bold green]✓[/bold green] Configuration saved to {output}")
    console.print("\nEdit the configuration file to customize settings:")
    console.print(f"  - Model: {config.ollama.model_name}")
    console.print(f"  - Embedding: {config.chromadb.embedding_model}")
    console.print(f"  - Database: {config.chromadb.persist_directory}")


@app.command()
def reset(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    config_path: Optional[str] = typer.Option(None, "--config", help="Path to config file")
):
    """Reset the vector database (WARNING: Deletes all indexed data)"""
    
    if not confirm:
        console.print("[bold red]WARNING:[/bold red] This will delete all indexed code!")
        response = typer.prompt("Are you sure? (yes/no)")
        if response.lower() != "yes":
            console.print("Operation cancelled")
            return
    
    config = load_config(config_path)
    rag = PrivaRepoRAG(config)
    
    rag.vector_store.reset_collection()
    console.print("[bold green]✓[/bold green] Database reset complete")


@app.command()
def export(
    output: str = typer.Argument(..., help="Output file path"),
    config_path: Optional[str] = typer.Option(None, "--config", help="Path to config file")
):
    """Export indexed chunks to JSON file"""
    
    config = load_config(config_path)
    rag = PrivaRepoRAG(config)
    
    rag.vector_store.export_chunks(output)


@app.command()
def import_data(
    input_file: str = typer.Argument(..., help="Input JSON file path"),
    config_path: Optional[str] = typer.Option(None, "--config", help="Path to config file")
):
    """Import chunks from JSON file"""
    
    config = load_config(config_path)
    rag = PrivaRepoRAG(config)
    
    rag.vector_store.import_chunks(input_file)


@app.command()
def version():
    """Show version information"""
    console.print("[bold]PrivaRepo - Local RAG for Code Intelligence[/bold]")
    console.print("Version: 1.0.0")
    console.print("\nFeatures:")
    console.print("  • Tree-sitter based code chunking (~40% precision improvement)")
    console.print("  • ChromaDB semantic search")
    console.print("  • Ollama local LLM (4-bit quantization)")
    console.print("  • RAGAS evaluation (0.92 faithfulness target)")
    console.print("  • Sub-2s query latency")


if __name__ == "__main__":
    app()
