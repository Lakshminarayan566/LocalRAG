"""
Ollama LLM interface for local code intelligence
Supports 4-bit quantization for sub-2s latency on resource-constrained environments
"""

import ollama
from typing import List, Dict, Optional, Generator
import time
from config import OllamaConfig


class OllamaLLM:
    """
    Local LLM interface using Ollama
    Optimized for code understanding and generation
    """
    
    def __init__(self, config: OllamaConfig):
        self.config = config
        self.client = ollama.Client(host=config.base_url)
        self._verify_model()
    
    def _verify_model(self):
        """Verify that the model is available"""
        try:
            # List available models
            models = self.client.list()
            available_models = [m['name'] for m in models.get('models', [])]
            
            if self.config.model_name not in available_models:
                print(f"Warning: Model {self.config.model_name} not found.")
                print(f"Available models: {available_models}")
                print(f"\nTo pull the model, run:")
                print(f"  ollama pull {self.config.model_name}")
            else:
                print(f"Model {self.config.model_name} is ready")
                
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            print(f"Make sure Ollama is running at {self.config.base_url}")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """
        Generate response from LLM
        
        Args:
            prompt: User prompt
            system_prompt: System instructions
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream response
            
        Returns:
            Generated text
        """
        if temperature is None:
            temperature = self.config.temperature
        if max_tokens is None:
            max_tokens = self.config.max_tokens
        
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})
        
        start_time = time.time()
        
        try:
            response = self.client.chat(
                model=self.config.model_name,
                messages=messages,
                options={
                    'temperature': temperature,
                    'top_p': self.config.top_p,
                    'num_predict': max_tokens,
                    'num_ctx': self.config.num_ctx,
                    'num_gpu': self.config.num_gpu,
                    'num_thread': self.config.num_thread,
                },
                stream=stream
            )
            
            if stream:
                return self._handle_stream(response)
            else:
                generated_text = response['message']['content']
                
        except Exception as e:
            print(f"Error generating response: {e}")
            generated_text = ""
        
        elapsed_time = time.time() - start_time
        print(f"Generation completed in {elapsed_time:.2f}s")
        
        return generated_text
    
    def _handle_stream(self, response) -> str:
        """Handle streaming response"""
        full_response = ""
        for chunk in response:
            content = chunk['message']['content']
            full_response += content
            print(content, end='', flush=True)
        print()  # New line after streaming
        return full_response
    
    def answer_code_question(
        self,
        question: str,
        context_chunks: List[Dict],
        include_code: bool = True
    ) -> str:
        """
        Answer a question about code using retrieved context
        
        Args:
            question: User's question
            context_chunks: Retrieved code chunks from vector store
            include_code: Whether to include code in response
            
        Returns:
            Answer to the question
        """
        # Build context from chunks
        context = self._build_context(context_chunks)
        
        # Create prompt
        system_prompt = """You are an expert code assistant. Your task is to answer questions about code accurately and concisely.

Guidelines:
- Base your answer on the provided code context
- Be precise and technical when needed
- Include code examples when relevant
- Cite specific functions or classes when referring to them
- If the context doesn't contain enough information, say so"""

        user_prompt = f"""Context (relevant code):
{context}

Question: {question}

Please provide a clear and accurate answer based on the code context above."""

        return self.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.1  # Low temperature for factual answers
        )
    
    def explain_code(self, code: str, language: str = "python") -> str:
        """Generate explanation for a code snippet"""
        system_prompt = f"You are an expert {language} developer. Explain the following code clearly and concisely."
        
        prompt = f"""Code to explain:
```{language}
{code}
```

Provide a clear explanation of:
1. What this code does
2. How it works
3. Any notable patterns or techniques used"""

        return self.generate(prompt=prompt, system_prompt=system_prompt)
    
    def generate_docstring(self, code: str, language: str = "python") -> str:
        """Generate documentation for code"""
        system_prompt = f"You are an expert at writing clear, comprehensive {language} documentation."
        
        prompt = f"""Generate a detailed docstring for this {language} code:

```{language}
{code}
```

Follow best practices for {language} documentation."""

        return self.generate(prompt=prompt, system_prompt=system_prompt)
    
    def find_bugs(self, code: str, language: str = "python") -> str:
        """Analyze code for potential bugs"""
        system_prompt = "You are an expert code reviewer. Identify potential bugs, issues, and improvements."
        
        prompt = f"""Review this {language} code for bugs and issues:

```{language}
{code}
```

Identify:
1. Potential bugs or errors
2. Edge cases not handled
3. Performance issues
4. Security concerns
5. Suggested improvements"""

        return self.generate(prompt=prompt, system_prompt=system_prompt)
    
    def suggest_improvements(self, code: str, language: str = "python") -> str:
        """Suggest code improvements"""
        system_prompt = "You are an expert software engineer. Suggest improvements for code quality, readability, and performance."
        
        prompt = f"""Analyze this {language} code and suggest improvements:

```{language}
{code}
```

Focus on:
1. Code quality and readability
2. Performance optimizations
3. Best practices
4. Design patterns"""

        return self.generate(prompt=prompt, system_prompt=system_prompt)
    
    def _build_context(self, chunks: List[Dict], max_chars: int = 4000) -> str:
        """
        Build context string from retrieved chunks
        
        Args:
            chunks: List of retrieved chunks
            max_chars: Maximum characters to include
            
        Returns:
            Formatted context string
        """
        context_parts = []
        current_chars = 0
        
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk['metadata']
            
            # Format chunk with metadata
            chunk_text = f"""
### Chunk {i} (Similarity: {chunk['similarity']:.3f})
File: {metadata['file_path']}
Type: {metadata['chunk_type']}
Lines: {metadata['start_line']}-{metadata['end_line']}
{f"Name: {metadata['name']}" if metadata.get('name') else ""}

```{metadata['language']}
{chunk['content']}
```
"""
            
            # Check if adding this chunk exceeds limit
            if current_chars + len(chunk_text) > max_chars:
                break
            
            context_parts.append(chunk_text)
            current_chars += len(chunk_text)
        
        return '\n'.join(context_parts)
    
    def get_model_info(self) -> Dict:
        """Get information about the current model"""
        try:
            model_info = self.client.show(self.config.model_name)
            return {
                'name': self.config.model_name,
                'size': model_info.get('size', 'unknown'),
                'quantization': self.config.quantization,
                'context_window': self.config.num_ctx,
                'family': model_info.get('family', 'unknown'),
            }
        except Exception as e:
            print(f"Error getting model info: {e}")
            return {'name': self.config.model_name, 'error': str(e)}
    
    def benchmark_latency(self, num_runs: int = 5) -> Dict:
        """
        Benchmark model latency
        
        Args:
            num_runs: Number of test runs
            
        Returns:
            Latency statistics
        """
        latencies = []
        test_prompt = "Write a simple Python function that adds two numbers."
        
        print(f"Running latency benchmark ({num_runs} runs)...")
        
        for i in range(num_runs):
            start = time.time()
            self.generate(test_prompt, max_tokens=100)
            latency = time.time() - start
            latencies.append(latency)
            print(f"Run {i+1}: {latency:.2f}s")
        
        return {
            'mean_latency': sum(latencies) / len(latencies),
            'min_latency': min(latencies),
            'max_latency': max(latencies),
            'num_runs': num_runs,
            'target_latency': 2.0,  # Sub-2s target from project description
            'meets_target': sum(latencies) / len(latencies) < 2.0
        }


class PromptTemplates:
    """Collection of prompt templates for common tasks"""
    
    CODE_EXPLANATION = """Explain the following {language} code in detail:

```{language}
{code}
```

Focus on:
- Overall purpose and functionality
- Key algorithms or patterns used
- Important implementation details
"""

    BUG_DETECTION = """Analyze this {language} code for potential issues:

```{language}
{code}
```

Look for:
- Logic errors
- Edge cases
- Performance problems
- Security vulnerabilities
"""

    CODE_REVIEW = """Perform a code review on this {language} code:

```{language}
{code}
```

Evaluate:
- Code quality and style
- Correctness
- Performance
- Maintainability
- Suggest specific improvements
"""

    FUNCTION_SEARCH = """Based on this codebase context, find functions that {description}:

Context:
{context}

Identify relevant functions and explain why they match.
"""
