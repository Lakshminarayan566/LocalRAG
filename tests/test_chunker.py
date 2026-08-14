"""
tests/test_chunker.py — Unit Tests for Tree-sitter Chunker

Tests AST parsing and chunk extraction for Python, Java, JavaScript, TypeScript.
Uses in-memory source strings (no filesystem required for most tests).
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from config import ChunkerConfig
from tree_sitter_chunker import (
    CodeChunk,
    JavaChunkExtractor,
    JavaScriptChunkExtractor,
    PythonChunkExtractor,
    TreeSitterChunker,
    _build_chunk_id,
    _build_document,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def chunker_config():
    return ChunkerConfig(min_chunk_lines=1, max_chunk_lines=500)


@pytest.fixture
def chunker(chunker_config):
    return TreeSitterChunker(chunker_config)


@pytest.fixture
def python_extractor():
    return PythonChunkExtractor()


@pytest.fixture
def java_extractor():
    return JavaChunkExtractor()


@pytest.fixture
def js_extractor():
    return JavaScriptChunkExtractor("javascript")


@pytest.fixture
def ts_extractor():
    return JavaScriptChunkExtractor("typescript")


# ---------------------------------------------------------------------------
# Helper: parse source with Tree-sitter
# ---------------------------------------------------------------------------

def parse_python(source: str):
    from tree_sitter_languages import get_parser
    parser = get_parser("python")
    return parser.parse(source.encode()), source.encode()


def parse_java(source: str):
    from tree_sitter_languages import get_parser
    parser = get_parser("java")
    return parser.parse(source.encode()), source.encode()


def parse_js(source: str):
    from tree_sitter_languages import get_parser
    parser = get_parser("javascript")
    return parser.parse(source.encode()), source.encode()


def parse_ts(source: str):
    from tree_sitter_languages import get_parser
    parser = get_parser("typescript")
    return parser.parse(source.encode()), source.encode()


# ---------------------------------------------------------------------------
# CodeChunk unit tests
# ---------------------------------------------------------------------------

class TestCodeChunk:
    def test_chunk_id_deterministic(self):
        """Same inputs must produce the same chunk ID."""
        id1 = _build_chunk_id("/path/to/file.py", 10, "def foo(): pass")
        id2 = _build_chunk_id("/path/to/file.py", 10, "def foo(): pass")
        assert id1 == id2

    def test_chunk_id_differs_on_different_content(self):
        id1 = _build_chunk_id("/path/file.py", 10, "def foo(): pass")
        id2 = _build_chunk_id("/path/file.py", 10, "def bar(): pass")
        assert id1 != id2

    def test_chunk_id_differs_on_different_line(self):
        id1 = _build_chunk_id("/path/file.py", 10, "def foo(): pass")
        id2 = _build_chunk_id("/path/file.py", 20, "def foo(): pass")
        assert id1 != id2

    def test_to_metadata_flat_dict(self):
        chunk = CodeChunk(
            chunk_id="abc123",
            code="def foo(): pass",
            document="FUNCTION foo",
            chunk_type="function",
            language="python",
            file_path="/src/main.py",
            start_line=5,
            end_line=6,
            function_name="foo",
            decorators=["@staticmethod"],
            parameters=["self", "x"],
        )
        meta = chunk.to_metadata()
        assert meta["chunk_type"] == "function"
        assert meta["language"] == "python"
        assert meta["function_name"] == "foo"
        assert meta["decorators"] == "@staticmethod"
        assert meta["parameters"] == "self,x"
        assert meta["line_count"] == 2
        # All values must be scalar (ChromaDB requirement)
        for key, val in meta.items():
            assert isinstance(val, (str, int, float, bool)), \
                f"Metadata value for '{key}' must be scalar, got {type(val)}"

    def test_build_document_includes_all_sections(self):
        chunk = CodeChunk(
            chunk_id="x",
            code="def foo(): pass",
            document="",
            chunk_type="function",
            language="python",
            file_path="/src/utils.py",
            start_line=1,
            end_line=1,
            function_name="foo",
            docstring="Computes foo.",
            class_name="MyClass",
        )
        doc = _build_document(chunk)
        assert "[FUNCTION]" in doc
        assert "[python]" in doc
        assert "MyClass" in doc
        assert "foo" in doc
        assert "/src/utils.py" in doc
        assert "Computes foo." in doc
        assert "def foo(): pass" in doc


# ---------------------------------------------------------------------------
# Python Extractor Tests
# ---------------------------------------------------------------------------

class TestPythonExtractor:
    def test_simple_function(self, python_extractor, chunker_config):
        source = textwrap.dedent("""\
            def greet(name: str) -> str:
                \"\"\"Return a greeting message.\"\"\"
                return f"Hello, {name}"
        """)
        tree, src_bytes = parse_python(source)
        chunks = python_extractor.extract(tree.root_node, src_bytes, "/test.py", chunker_config)

        functions = [c for c in chunks if c.chunk_type == "function"]
        assert len(functions) >= 1
        fn = functions[0]
        assert fn.function_name == "greet"
        assert fn.language == "python"
        assert fn.docstring_exists
        assert "name" in fn.parameters or "name: str" in fn.parameters
        assert fn.return_type is not None

    def test_class_with_methods(self, python_extractor, chunker_config):
        source = textwrap.dedent("""\
            class Animal(Base):
                \"\"\"Base animal class.\"\"\"

                def __init__(self, name: str):
                    self.name = name

                def speak(self) -> str:
                    return f"{self.name} makes a sound"

                @staticmethod
                def category() -> str:
                    return "Animal"
        """)
        tree, src_bytes = parse_python(source)
        chunks = python_extractor.extract(tree.root_node, src_bytes, "/test.py", chunker_config)

        class_chunks = [c for c in chunks if c.chunk_type == "class"]
        method_chunks = [c for c in chunks if c.chunk_type == "method"]

        assert len(class_chunks) >= 1
        cls = class_chunks[0]
        assert cls.class_name == "Animal"
        assert cls.parent_class == "Base"
        assert cls.docstring_exists

        assert len(method_chunks) >= 2
        method_names = {m.function_name for m in method_chunks}
        assert "__init__" in method_names
        assert "speak" in method_names

    def test_imports_extracted(self, python_extractor, chunker_config):
        source = textwrap.dedent("""\
            import os
            import sys
            from pathlib import Path
            from typing import List, Optional

            def main():
                pass
        """)
        tree, src_bytes = parse_python(source)
        chunks = python_extractor.extract(tree.root_node, src_bytes, "/test.py", chunker_config)

        imports = [c for c in chunks if c.chunk_type == "imports"]
        assert len(imports) == 1
        assert "import os" in imports[0].code

    def test_async_function_detected(self, python_extractor, chunker_config):
        source = textwrap.dedent("""\
            async def fetch_data(url: str) -> dict:
                \"\"\"Fetch data from URL.\"\"\"
                return {}
        """)
        tree, src_bytes = parse_python(source)
        chunks = python_extractor.extract(tree.root_node, src_bytes, "/test.py", chunker_config)

        functions = [c for c in chunks if c.chunk_type == "function"]
        assert any(f.is_async for f in functions)

    def test_no_duplicate_chunk_ids(self, python_extractor, chunker_config):
        source = textwrap.dedent("""\
            class Foo:
                def bar(self): pass
                def baz(self): pass

            def standalone(): pass
        """)
        tree, src_bytes = parse_python(source)
        chunks = python_extractor.extract(tree.root_node, src_bytes, "/test.py", chunker_config)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"

    def test_min_chunk_lines_filter(self, chunker_config):
        """Chunks below min_chunk_lines should be filtered out."""
        config = ChunkerConfig(min_chunk_lines=5)
        extractor = PythonChunkExtractor()
        source = "def tiny(): pass\n"
        tree, src_bytes = parse_python(source)
        chunks = extractor.extract(tree.root_node, src_bytes, "/test.py", config)
        # The single-line function should be filtered
        functions = [c for c in chunks if c.chunk_type == "function"]
        assert len(functions) == 0


# ---------------------------------------------------------------------------
# Java Extractor Tests
# ---------------------------------------------------------------------------

class TestJavaExtractor:
    def test_simple_class_and_method(self, java_extractor, chunker_config):
        source = textwrap.dedent("""\
            package com.example;

            import java.util.List;

            public class UserService extends BaseService {
                /**
                 * Get user by ID.
                 */
                public User getUserById(Long id) {
                    return repository.findById(id);
                }

                private void validateUser(User user) {
                    if (user == null) throw new IllegalArgumentException();
                }
            }
        """)
        tree, src_bytes = parse_java(source)
        chunks = java_extractor.extract(tree.root_node, src_bytes, "/UserService.java", chunker_config)

        classes = [c for c in chunks if c.chunk_type == "class"]
        methods = [c for c in chunks if c.chunk_type == "method"]

        assert len(classes) >= 1
        assert classes[0].class_name == "UserService"
        assert classes[0].parent_class == "BaseService"

        method_names = {m.function_name for m in methods}
        assert "getUserById" in method_names
        assert "validateUser" in method_names

    def test_imports_extracted_java(self, java_extractor, chunker_config):
        source = textwrap.dedent("""\
            import java.util.List;
            import java.util.Map;
            import org.springframework.stereotype.Service;

            public class Foo {}
        """)
        tree, src_bytes = parse_java(source)
        chunks = java_extractor.extract(tree.root_node, src_bytes, "/Foo.java", chunker_config)

        imports = [c for c in chunks if c.chunk_type == "imports"]
        assert len(imports) >= 1

    def test_interface_extracted(self, java_extractor, chunker_config):
        source = textwrap.dedent("""\
            public interface Repository<T, ID> {
                T findById(ID id);
                List<T> findAll();
            }
        """)
        tree, src_bytes = parse_java(source)
        chunks = java_extractor.extract(tree.root_node, src_bytes, "/Repo.java", chunker_config)
        classes = [c for c in chunks if c.chunk_type == "class"]
        assert len(classes) >= 1


# ---------------------------------------------------------------------------
# JavaScript / TypeScript Extractor Tests
# ---------------------------------------------------------------------------

class TestJavaScriptExtractor:
    def test_function_declaration(self, js_extractor, chunker_config):
        source = textwrap.dedent("""\
            /**
             * Add two numbers.
             */
            function add(a, b) {
                return a + b;
            }

            const multiply = (a, b) => a * b;
        """)
        tree, src_bytes = parse_js(source)
        chunks = js_extractor.extract(tree.root_node, src_bytes, "/math.js", chunker_config)

        functions = [c for c in chunks if c.chunk_type == "function"]
        assert any(f.function_name == "add" for f in functions)

    def test_class_with_methods_js(self, js_extractor, chunker_config):
        source = textwrap.dedent("""\
            class EventEmitter extends BaseEmitter {
                constructor() {
                    super();
                    this.listeners = {};
                }

                on(event, callback) {
                    this.listeners[event] = callback;
                }

                emit(event, data) {
                    if (this.listeners[event]) {
                        this.listeners[event](data);
                    }
                }
            }
        """)
        tree, src_bytes = parse_js(source)
        chunks = js_extractor.extract(tree.root_node, src_bytes, "/emitter.js", chunker_config)

        classes = [c for c in chunks if c.chunk_type == "class"]
        assert len(classes) >= 1
        assert classes[0].class_name == "EventEmitter"
        assert classes[0].parent_class == "BaseEmitter"

    def test_typescript_type_annotations(self, ts_extractor, chunker_config):
        source = textwrap.dedent("""\
            interface User {
                id: number;
                name: string;
            }

            function fetchUser(id: number): Promise<User> {
                return fetch(`/api/users/${id}`).then(r => r.json());
            }

            class UserService {
                private users: User[] = [];

                addUser(user: User): void {
                    this.users.push(user);
                }
            }
        """)
        tree, src_bytes = parse_ts(source)
        chunks = ts_extractor.extract(tree.root_node, src_bytes, "/user.ts", chunker_config)

        assert len(chunks) > 0
        languages = {c.language for c in chunks}
        assert "typescript" in languages


# ---------------------------------------------------------------------------
# TreeSitterChunker Integration Tests
# ---------------------------------------------------------------------------

class TestTreeSitterChunker:
    def test_chunk_file_python(self, chunker, tmp_path):
        py_file = tmp_path / "sample.py"
        py_file.write_text(textwrap.dedent("""\
            import os
            from pathlib import Path

            class FileProcessor:
                def process(self, path: str) -> None:
                    \"\"\"Process a file.\"\"\"
                    data = Path(path).read_text()
                    return data.strip()

            def standalone_helper(x: int) -> int:
                return x * 2
        """))

        chunks = chunker.chunk_file(str(py_file))
        assert len(chunks) > 0

        types = {c.chunk_type for c in chunks}
        assert "class" in types
        assert "method" in types
        assert "function" in types or "imports" in types

    def test_chunk_file_unsupported_extension(self, chunker, tmp_path):
        txt_file = tmp_path / "readme.txt"
        txt_file.write_text("This is plain text.")
        chunks = chunker.chunk_file(str(txt_file))
        assert chunks == []

    def test_chunk_file_missing_file(self, chunker):
        chunks = chunker.chunk_file("/nonexistent/path/file.py")
        assert chunks == []

    def test_chunk_repository(self, chunker, tmp_path):
        # Create a minimal repo structure
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text(textwrap.dedent("""\
            def main():
                print("Hello, World!")

            class App:
                def run(self):
                    main()
        """))
        (tmp_path / "src" / "utils.py").write_text(textwrap.dedent("""\
            def helper(x):
                return x + 1
        """))

        chunks = chunker.chunk_repository(str(tmp_path))
        assert len(chunks) > 0

        files = {c.file_path for c in chunks}
        assert any("main.py" in f for f in files)
        assert any("utils.py" in f for f in files)

    def test_chunk_repository_deduplication(self, chunker, tmp_path):
        """Indexing same repo twice should not duplicate chunk IDs."""
        (tmp_path / "test.py").write_text("def foo(): pass\ndef bar(): pass\n")

        chunks1 = chunker.chunk_repository(str(tmp_path))
        chunks2 = chunker.chunk_repository(str(tmp_path))

        ids1 = {c.chunk_id for c in chunks1}
        ids2 = {c.chunk_id for c in chunks2}
        assert ids1 == ids2, "Same repo produces different chunk IDs on second run"

    def test_chunk_repository_excludes_patterns(self, chunker, tmp_path):
        (tmp_path / "src.py").write_text("def real(): pass")
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "cached.py").write_text("def cached(): pass")

        chunks = chunker.chunk_repository(str(tmp_path))
        files = {c.file_path for c in chunks}
        assert not any("__pycache__" in f for f in files)

    def test_language_detection(self, chunker):
        assert chunker.get_language_for_file("main.py") == "python"
        assert chunker.get_language_for_file("App.java") == "java"
        assert chunker.get_language_for_file("index.js") == "javascript"
        assert chunker.get_language_for_file("types.ts") == "typescript"
        assert chunker.get_language_for_file("readme.md") is None


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_file(self, chunker, tmp_path):
        (tmp_path / "empty.py").write_text("")
        chunks = chunker.chunk_file(str(tmp_path / "empty.py"))
        assert isinstance(chunks, list)

    def test_file_with_only_comments(self, chunker, tmp_path):
        (tmp_path / "comments.py").write_text("# This file has only comments\n# No actual code\n")
        chunks = chunker.chunk_file(str(tmp_path / "comments.py"))
        assert isinstance(chunks, list)

    def test_unicode_source(self, chunker, tmp_path):
        py_file = tmp_path / "unicode.py"
        py_file.write_text(
            '# -*- coding: utf-8 -*-\ndef grüßen(name: str) -> str:\n    return f"Hallo, {name}! 🎉"\n',
            encoding="utf-8",
        )
        chunks = chunker.chunk_file(str(py_file))
        assert isinstance(chunks, list)

    def test_deeply_nested_class(self, chunker, tmp_path):
        (tmp_path / "nested.py").write_text(textwrap.dedent("""\
            class Outer:
                class Inner:
                    def method(self):
                        class InnerInner:
                            pass
                        return InnerInner()
        """))
        chunks = chunker.chunk_file(str(tmp_path / "nested.py"))
        assert isinstance(chunks, list)
        assert len(chunks) > 0
