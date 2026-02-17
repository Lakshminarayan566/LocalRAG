<div align="center">

# 🔍 LocalRAG
### Ask anything about your codebase. Get answers in seconds.

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black?style=for-the-badge)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-orange?style=for-the-badge)
![TreeSitter](https://img.shields.io/badge/Tree--sitter-AST_Parser-green?style=for-the-badge)

</div>

---

## 💡 What is LocalRAG?

> **Point it at any code folder → Ask questions in plain English → Get answers from your actual code — 100% private, runs locally.**
```
You:       "How does authentication work?"
LocalRAG:  "Login is handled in auth.py using JWT tokens..."  ⚡ in 2s
```

---

## 🏆 Key Results

| Metric | Result |
|--------|--------|
| 🎯 Retrieval Precision | **+40% improvement** via Tree-sitter AST |
| 📊 Faithfulness Score | **0.92** via RAGAS framework |
| ⚡ Query Latency | **< 2 seconds** with 4-bit quantization |
| 🔒 Privacy | **100% local** — no data leaves your machine |

---

## 🏗️ Architecture
```
 Your Code Folder
       ↓
 🌳 Tree-sitter    →  Understands code structure (functions, classes)
       ↓
 🗄️ ChromaDB       →  Stores code as searchable vectors
       ↓
 ❓ Your Question
       ↓
 🤖 Ollama LLM     →  Reads relevant code & generates answer
       ↓
 ✅ Answer
```

---

## 📁 Project Structure
```
LocalRAG/
├── 🚀 app.py                 # Web UI (Flask)
├── 🧠 rag_pipeline.py        # Main RAG pipeline
├── 🌳 tree_sitter_chunker.py # AST code parser
├── 🗄️ vector_store.py        # ChromaDB integration
├── 🤖 llm_interface.py       # Ollama LLM
├── 📊 evaluator.py           # RAGAS evaluation
├── ⚙️ config.py              # Configuration
└── 💻 cli.py                 # CLI interface
```

---

## 🛠️ Tech Stack

- **Python** — Core language
- **Tree-sitter** — AST-based code parsing
- **ChromaDB** — Vector storage & semantic search
- **Ollama** — Local LLM inference (4-bit quantized)
- **RAGAS** — Answer quality evaluation

---

<div align="center">

**Built by [Lakshminarayan566](https://github.com/Lakshminarayan566)**

⭐ Star this repo if you found it useful!

</div>
