"""
RAGAS Evaluation Framework
Quantifies model faithfulness and retrieval quality
Target: 0.92 faithfulness score
"""

from typing import List, Dict, Optional
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from rich.console import Console
from rich.table import Table

from config import RAGASConfig
from rag_pipeline import PrivaRepoRAG


console = Console()


class RAGASEvaluator:
    """
    Evaluate RAG system using RAGAS framework
    Measures: faithfulness, answer relevancy, context precision, context recall
    """
    
    def __init__(self, rag_system: PrivaRepoRAG, config: Optional[RAGASConfig] = None):
        self.rag_system = rag_system
        self.config = config or RAGASConfig()
        
        # Select metrics to compute
        self.metrics = []
        if self.config.compute_faithfulness:
            self.metrics.append(faithfulness)
        if self.config.compute_answer_relevancy:
            self.metrics.append(answer_relevancy)
        if self.config.compute_context_precision:
            self.metrics.append(context_precision)
        if self.config.compute_context_recall:
            self.metrics.append(context_recall)
    
    def create_evaluation_dataset(
        self,
        test_cases: List[Dict],
        include_ground_truth: bool = True
    ) -> Dataset:
        """
        Create evaluation dataset from test cases
        
        Args:
            test_cases: List of test cases with questions and optionally ground truth
            include_ground_truth: Whether test cases include ground truth answers
            
        Returns:
            RAGAS-compatible dataset
        """
        eval_data = {
            'question': [],
            'answer': [],
            'contexts': [],
            'ground_truth': [] if include_ground_truth else None
        }
        
        for test_case in test_cases:
            question = test_case['question']
            
            # Get RAG response
            result = self.rag_system.query(question, include_context=True)
            
            # Extract contexts from retrieved chunks
            contexts = [
                chunk['content'] for chunk in result['retrieved_chunks']
            ]
            
            eval_data['question'].append(question)
            eval_data['answer'].append(result['answer'])
            eval_data['contexts'].append(contexts)
            
            if include_ground_truth:
                eval_data['ground_truth'].append(test_case.get('ground_truth', ''))
        
        # Create dataset
        if not include_ground_truth:
            del eval_data['ground_truth']
        
        return Dataset.from_dict(eval_data)
    
    def evaluate(
        self,
        test_cases: List[Dict],
        include_ground_truth: bool = True
    ) -> Dict:
        """
        Run RAGAS evaluation
        
        Args:
            test_cases: Test cases with questions and ground truth
            include_ground_truth: Whether ground truth is available
            
        Returns:
            Evaluation results
        """
        console.print("\n[bold]Running RAGAS Evaluation...[/bold]\n")
        
        # Create dataset
        console.print("[dim]Creating evaluation dataset...[/dim]")
        dataset = self.create_evaluation_dataset(test_cases, include_ground_truth)
        
        console.print(f"[dim]Evaluating {len(test_cases)} test cases...[/dim]")
        
        # Run evaluation
        results = evaluate(
            dataset=dataset,
            metrics=self.metrics,
        )
        
        # Process results
        scores = {
            'overall': results,
            'per_metric': {}
        }
        
        # Extract individual metric scores
        if self.config.compute_faithfulness:
            scores['per_metric']['faithfulness'] = results['faithfulness']
        if self.config.compute_answer_relevancy:
            scores['per_metric']['answer_relevancy'] = results['answer_relevancy']
        if self.config.compute_context_precision:
            scores['per_metric']['context_precision'] = results['context_precision']
        if self.config.compute_context_recall:
            scores['per_metric']['context_recall'] = results['context_recall']
        
        # Display results
        self._display_results(scores)
        
        # Check if faithfulness target is met
        if self.config.compute_faithfulness:
            faithfulness_score = scores['per_metric']['faithfulness']
            target_met = faithfulness_score >= self.config.target_faithfulness
            
            console.print(f"\n[bold]Faithfulness Target ({self.config.target_faithfulness}):[/bold] ", end="")
            if target_met:
                console.print(f"[bold green]✓ Met ({faithfulness_score:.3f})[/bold green]")
            else:
                console.print(f"[bold red]✗ Not met ({faithfulness_score:.3f})[/bold red]")
        
        return scores
    
    def _display_results(self, scores: Dict):
        """Display evaluation results in a formatted table"""
        table = Table(title="RAGAS Evaluation Results", show_header=True)
        
        table.add_column("Metric", style="cyan")
        table.add_column("Score", style="magenta")
        table.add_column("Status", style="green")
        
        for metric_name, score in scores['per_metric'].items():
            # Determine status
            if metric_name == 'faithfulness':
                target = self.config.target_faithfulness
                status = "✓ Target Met" if score >= target else "✗ Below Target"
            else:
                status = "✓ Good" if score >= 0.7 else "⚠ Needs Improvement"
            
            table.add_row(
                metric_name.replace('_', ' ').title(),
                f"{score:.3f}",
                status
            )
        
        console.print(table)
    
    def evaluate_retrieval_precision(
        self,
        test_cases: List[Dict],
        baseline_top_k: int = 5
    ) -> Dict:
        """
        Evaluate retrieval precision improvement
        Target: ~40% improvement over baseline
        
        Args:
            test_cases: Test cases with questions and relevant chunk IDs
            baseline_top_k: Number of chunks to retrieve
            
        Returns:
            Precision metrics
        """
        console.print("\n[bold]Evaluating Retrieval Precision...[/bold]\n")
        
        precisions = []
        recalls = []
        
        for test_case in test_cases:
            question = test_case['question']
            relevant_chunks = set(test_case.get('relevant_chunk_ids', []))
            
            if not relevant_chunks:
                continue
            
            # Retrieve chunks
            retrieved = self.rag_system.vector_store.search(
                query=question,
                top_k=baseline_top_k
            )
            
            retrieved_ids = set(chunk['chunk_id'] for chunk in retrieved)
            
            # Calculate precision and recall
            true_positives = len(retrieved_ids & relevant_chunks)
            precision = true_positives / len(retrieved_ids) if retrieved_ids else 0
            recall = true_positives / len(relevant_chunks) if relevant_chunks else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        avg_precision = sum(precisions) / len(precisions) if precisions else 0
        avg_recall = sum(recalls) / len(recalls) if recalls else 0
        
        # Calculate F1 score
        f1_score = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        results = {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': f1_score,
            'num_test_cases': len(precisions),
            'target_improvement': self.config.target_precision_improvement,
        }
        
        console.print(f"Average Precision: {avg_precision:.3f}")
        console.print(f"Average Recall: {avg_recall:.3f}")
        console.print(f"F1 Score: {f1_score:.3f}")
        console.print(f"\n[dim]Target: {self.config.target_precision_improvement*100:.0f}% improvement over baseline[/dim]")
        
        return results
    
    def create_test_suite(self) -> List[Dict]:
        """
        Create a standard test suite for code intelligence
        
        Returns:
            List of test cases
        """
        test_suite = [
            {
                'question': 'How does the authentication system work?',
                'ground_truth': 'The authentication system uses JWT tokens with bcrypt password hashing.',
                'category': 'architecture'
            },
            {
                'question': 'Find the database connection initialization code',
                'ground_truth': 'Database connection is initialized in db.py using SQLAlchemy.',
                'category': 'infrastructure'
            },
            {
                'question': 'What are the main API endpoints?',
                'ground_truth': 'Main endpoints include /api/users, /api/auth, and /api/data.',
                'category': 'api'
            },
            {
                'question': 'Show me error handling patterns',
                'ground_truth': 'Error handling uses try-except blocks with custom exception classes.',
                'category': 'patterns'
            },
            {
                'question': 'How is data validation performed?',
                'ground_truth': 'Data validation uses Pydantic models with custom validators.',
                'category': 'validation'
            },
            {
                'question': 'Find test utility functions',
                'ground_truth': 'Test utilities are in conftest.py and include fixtures for database and auth.',
                'category': 'testing'
            },
            {
                'question': 'What caching strategy is used?',
                'ground_truth': 'Redis caching with TTL-based expiration for frequently accessed data.',
                'category': 'performance'
            },
            {
                'question': 'How are configuration settings managed?',
                'ground_truth': 'Settings are managed using environment variables and pydantic-settings.',
                'category': 'configuration'
            },
        ]
        
        return test_suite
    
    def run_comprehensive_evaluation(self) -> Dict:
        """
        Run comprehensive evaluation with standard test suite
        
        Returns:
            Complete evaluation results
        """
        console.print("\n[bold blue]Running Comprehensive Evaluation[/bold blue]")
        
        # Create test suite
        test_suite = self.create_test_suite()
        
        # Run RAGAS evaluation
        ragas_results = self.evaluate(test_suite, include_ground_truth=True)
        
        # Run precision evaluation
        # Note: For precision, you'd need to manually annotate relevant chunks
        # This is a placeholder for the concept
        # precision_results = self.evaluate_retrieval_precision(test_suite)
        
        results = {
            'ragas_metrics': ragas_results,
            'test_suite_size': len(test_suite),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        return results
    
    def export_results(self, results: Dict, output_path: str):
        """Export evaluation results to file"""
        import json
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        console.print(f"\n[bold green]Results exported to {output_path}[/bold green]")
    
    def compare_with_baseline(
        self,
        test_cases: List[Dict],
        baseline_results: Dict
    ) -> Dict:
        """
        Compare current system with baseline
        
        Args:
            test_cases: Test cases
            baseline_results: Previous evaluation results
            
        Returns:
            Comparison metrics
        """
        current_results = self.evaluate(test_cases)
        
        comparison = {
            'current': current_results,
            'baseline': baseline_results,
            'improvements': {}
        }
        
        # Calculate improvements
        for metric in current_results['per_metric']:
            current_score = current_results['per_metric'][metric]
            baseline_score = baseline_results['per_metric'].get(metric, 0)
            
            if baseline_score > 0:
                improvement = (current_score - baseline_score) / baseline_score
                comparison['improvements'][metric] = {
                    'absolute': current_score - baseline_score,
                    'relative': improvement,
                    'percentage': improvement * 100
                }
        
        # Display comparison
        console.print("\n[bold]Comparison with Baseline:[/bold]")
        for metric, improvement in comparison['improvements'].items():
            console.print(f"{metric}: {improvement['percentage']:+.1f}%")
        
        return comparison
