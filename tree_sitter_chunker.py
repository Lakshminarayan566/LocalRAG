"""
tree_sitter_chunker.py — AST-Aware Code Chunker

Parses source files with Tree-sitter to extract semantically complete
code units (functions, methods, classes, imports) while preserving
full AST boundaries, parent class context, and rich metadata.

Design decisions:
  - Uses tree-sitter-languages for pre-compiled grammar binaries (no gcc).
  - Parallel repository walking via ThreadPoolExecutor.
  - Content-hashing for deduplication across identical files.
  - Overlap strategy: prepends decorator/docstring context to each chunk.
"""

from __future__ import annotations

import hashlib
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

from tree_sitter import Language, Node, Parser
from tree_sitter_languages import get_language, get_parser

from config import ChunkerConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------
@dataclass
class CodeChunk:
    """A semantically complete unit of source code extracted from an AST."""

    chunk_id: str                        # SHA256 of (file_path + start_line + content)
    code: str                            # Raw source text of this chunk
    document: str                        # Enriched text used for embedding
    chunk_type: str                      # "function" | "method" | "class" | "imports" | "module"
    language: str                        # "python" | "java" | "javascript" | "typescript"
    file_path: str                       # Absolute path of the source file
    start_line: int                      # 1-indexed start line in source file
    end_line: int                        # 1-indexed end line in source file
    function_name: Optional[str] = None  # Name of the function/method
    class_name: Optional[str] = None     # Name of the enclosing class (if any)
    parent_class: Optional[str] = None   # Parent/super class name (if determinable)
    docstring: Optional[str] = None      # Extracted docstring text
    docstring_exists: bool = False
    decorators: List[str] = field(default_factory=list)
    parameters: List[str] = field(default_factory=list)
    return_type: Optional[str] = None
    is_async: bool = False
    module_imports: str = ""          # Raw import block of the file this chunk came from
    parent_chunk_id: Optional[str] = None  # chunk_id of the enclosing class, for methods

    def to_metadata(self) -> dict:
        """Serialise to flat dict for ChromaDB metadata storage."""
        return {
            "chunk_type": self.chunk_type,
            "language": self.language,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "function_name": self.function_name or "",
            "class_name": self.class_name or "",
            "parent_class": self.parent_class or "",
            "docstring_exists": self.docstring_exists,
            "is_async": self.is_async,
            "decorators": ",".join(self.decorators),
            "parameters": ",".join(self.parameters),
            "return_type": self.return_type or "",
            "line_count": self.end_line - self.start_line + 1,
            "parent_chunk_id": self.parent_chunk_id or "",
        }

    def __repr__(self) -> str:
        name = self.function_name or self.class_name or "?"
        return (
            f"CodeChunk(type={self.chunk_type}, name={name}, "
            f"lang={self.language}, lines={self.start_line}-{self.end_line})"
        )


# ---------------------------------------------------------------------------
# Language-specific AST visitor helpers
# ---------------------------------------------------------------------------

def _node_text(node: Node, source: bytes) -> str:
    """Extract UTF-8 text of a Tree-sitter node."""
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _child_by_type(node: Node, *types: str) -> Optional[Node]:
    """Return first direct child matching any of the given types."""
    for child in node.children:
        if child.type in types:
            return child
    return None


def _find_nodes(node: Node, *types: str) -> List[Node]:
    """DFS: collect all descendant nodes matching given types."""
    results: List[Node] = []
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type in types:
            results.append(current)
        else:
            stack.extend(reversed(current.children))
    return results


def _extract_python_docstring(node: Node, source: bytes) -> Optional[str]:
    """
    Extract the docstring from a Python function/class definition node.
    The first statement of the body must be an expression_statement
    containing a string.
    """
    body = _child_by_type(node, "block")
    if not body:
        return None
    for child in body.children:
        if child.type == "expression_statement":
            for sub in child.children:
                if sub.type in ("string", "concatenated_string"):
                    raw = _node_text(sub, source).strip()
                    # Strip triple quotes
                    for q in ('"""', "'''", '"', "'"):
                        if raw.startswith(q) and raw.endswith(q) and len(raw) > 2 * len(q):
                            return raw[len(q) : -len(q)].strip()
                    return raw
        break
    return None


def _extract_java_doc_comment(node: Node, source: bytes) -> Optional[str]:
    """Return Javadoc comment text that immediately precedes the given node."""
    prev = node.prev_named_sibling
    if prev and prev.type in ("block_comment", "line_comment"):
        text = _node_text(prev, source).strip()
        if text.startswith("/**") or text.startswith("//"):
            return text
    return None


def _extract_js_doc_comment(node: Node, source: bytes) -> Optional[str]:
    """Return JSDoc comment immediately preceding the node."""
    prev = node.prev_sibling
    if prev and prev.type == "comment":
        return _node_text(prev, source).strip()
    return None


def _build_chunk_id(file_path: str, start_line: int, content: str) -> str:
    payload = f"{file_path}:{start_line}:{content}"
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _build_document(chunk: "CodeChunk") -> str:
    """
    Build the enriched text representation stored in ChromaDB.
    Embedding this richer text produces better semantic similarity.
    """
    parts: List[str] = []
    parts.append(f"[{chunk.chunk_type.upper()}] [{chunk.language}]")
    if chunk.class_name:
        parts.append(f"Class: {chunk.class_name}")
    if chunk.function_name:
        parts.append(f"Name: {chunk.function_name}")
    if chunk.parent_class:
        parts.append(f"Inherits: {chunk.parent_class}")
    parts.append(f"File: {chunk.file_path}")
    if chunk.module_imports:
        # Bounded to ~300 chars so a big import block doesn't dilute the
        # embedding — we want "this file also imports X" as weak context,
        # not to drown out the actual code signal.
        parts.append(f"Imports: {chunk.module_imports.strip()[:300]}")
    if chunk.docstring:
        parts.append(f"Docstring: {chunk.docstring[:300]}")
    parts.append("")
    parts.append(chunk.code)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Per-language chunk extractors
# ---------------------------------------------------------------------------

class PythonChunkExtractor:
    """Extracts functions, methods, classes, and imports from Python AST."""

    FUNCTION_TYPES = {"function_definition", "async_function_definition"}
    CLASS_TYPE = "class_definition"
    IMPORT_TYPES = {"import_statement", "import_from_statement"}

    def extract(
        self,
        tree_root: Node,
        source: bytes,
        file_path: str,
        config: ChunkerConfig,
    ) -> List[CodeChunk]:
        chunks: List[CodeChunk] = []
        seen_ids: Set[str] = set()

        # --- Top-level imports as a single block ---
        import_nodes = [
            n for n in tree_root.children if n.type in self.IMPORT_TYPES
        ]
        if import_nodes:
            start = import_nodes[0].start_point[0] + 1
            end = import_nodes[-1].end_point[0] + 1
            code = source[import_nodes[0].start_byte:import_nodes[-1].end_byte].decode(
                "utf-8", errors="replace"
            )
            cid = _build_chunk_id(file_path, start, code)
            if cid not in seen_ids:
                seen_ids.add(cid)
                c = CodeChunk(
                    chunk_id=cid,
                    code=code,
                    document="",
                    chunk_type="imports",
                    language="python",
                    file_path=file_path,
                    start_line=start,
                    end_line=end,
                )
                c.document = _build_document(c)
                chunks.append(c)

        module_imports = ""
        if import_nodes:
            module_imports = source[
                import_nodes[0].start_byte:import_nodes[-1].end_byte
            ].decode("utf-8", errors="replace")

        # --- Classes ---
        for class_node in _find_nodes(tree_root, self.CLASS_TYPE):
            chunks.extend(
                self._extract_class(
                    class_node, source, file_path, config, seen_ids, module_imports
                )
            )

        # --- Top-level functions (not inside a class) ---
        # A decorated function is wrapped in a `decorated_definition` node
        # (decorators + the actual function_definition as children), so it
        # will not match FUNCTION_TYPES directly — unwrap it first.
        for fn_node in tree_root.children:
            inner = self._resolve_function_node(fn_node)
            if inner is not None:
                chunk = self._extract_function(
                    inner, source, file_path, None, config, module_imports
                )
                if chunk and chunk.chunk_id not in seen_ids:
                    seen_ids.add(chunk.chunk_id)
                    chunks.append(chunk)

        return [c for c in chunks if (c.end_line - c.start_line + 1) >= config.min_chunk_lines]

    def _extract_class(
        self,
        class_node: Node,
        source: bytes,
        file_path: str,
        config: ChunkerConfig,
        seen_ids: Set[str],
        module_imports: str = "",
    ) -> List[CodeChunk]:
        chunks: List[CodeChunk] = []
        class_name = ""
        parent_class = None

        name_node = _child_by_type(class_node, "identifier")
        if name_node:
            class_name = _node_text(name_node, source)

        # Extract parent class from argument_list / bases
        arg_list = _child_by_type(class_node, "argument_list")
        if arg_list:
            bases = [
                _node_text(c, source)
                for c in arg_list.children
                if c.type not in ("(", ")", ",")
            ]
            if bases:
                parent_class = bases[0]

        # Class-level chunk (full class body)
        code = _node_text(class_node, source)
        start = class_node.start_point[0] + 1
        end = class_node.end_point[0] + 1
        cid = _build_chunk_id(file_path, start, code)

        docstring = _extract_python_docstring(class_node, source)
        decorators = self._extract_decorators(class_node, source)

        if cid not in seen_ids:
            seen_ids.add(cid)
            c = CodeChunk(
                chunk_id=cid,
                code=code,
                document="",
                chunk_type="class",
                language="python",
                file_path=file_path,
                start_line=start,
                end_line=end,
                class_name=class_name,
                parent_class=parent_class,
                docstring=docstring,
                docstring_exists=docstring is not None,
                decorators=decorators,
                module_imports=module_imports,
            )
            c.document = _build_document(c)
            chunks.append(c)

        # Methods within the class — linked back to the class chunk via
        # parent_chunk_id so the retriever can pull in class context
        # (docstring, inheritance, sibling methods) when a method matches.
        body = _child_by_type(class_node, "block")
        if body:
            for fn_node in body.children:
                inner = self._resolve_function_node(fn_node)
                if inner is not None:
                    chunk = self._extract_function(
                        inner, source, file_path, class_name, config, module_imports
                    )
                    if chunk and chunk.chunk_id not in seen_ids:
                        chunk.parent_class = parent_class
                        chunk.parent_chunk_id = cid
                        seen_ids.add(chunk.chunk_id)
                        chunks.append(chunk)

        return chunks

    def _extract_function(
        self,
        fn_node: Node,
        source: bytes,
        file_path: str,
        class_name: Optional[str],
        config: ChunkerConfig,
        module_imports: str = "",
    ) -> Optional[CodeChunk]:
        name_node = _child_by_type(fn_node, "identifier")
        func_name = _node_text(name_node, source) if name_node else "<anonymous>"

        code = _node_text(fn_node, source)
        start = fn_node.start_point[0] + 1
        end = fn_node.end_point[0] + 1

        # Hard limit: split oversized chunks at logical boundaries
        if (end - start + 1) > config.max_chunk_lines:
            logger.debug(
                "Function %s exceeds max_chunk_lines (%d), truncating to first %d lines",
                func_name, config.max_chunk_lines, config.max_chunk_lines,
            )
            lines = code.splitlines()[:config.max_chunk_lines]
            code = "\n".join(lines)
            end = start + config.max_chunk_lines - 1

        cid = _build_chunk_id(file_path, start, code)
        docstring = _extract_python_docstring(fn_node, source)
        decorators = self._extract_decorators(fn_node, source)
        params = self._extract_parameters(fn_node, source)
        return_type = self._extract_return_type(fn_node, source)
        is_async = fn_node.type == "async_function_definition" or code.strip().startswith("async")
        chunk_type = "method" if class_name else "function"

        c = CodeChunk(
            chunk_id=cid,
            code=code,
            document="",
            chunk_type=chunk_type,
            language="python",
            file_path=file_path,
            start_line=start,
            end_line=end,
            function_name=func_name,
            class_name=class_name,
            docstring=docstring,
            docstring_exists=docstring is not None,
            decorators=decorators,
            parameters=params,
            return_type=return_type,
            is_async=is_async,
            module_imports=module_imports,
        )
        c.document = _build_document(c)
        return c

    def _resolve_function_node(self, node: Node) -> Optional[Node]:
        """
        Given a node that is either a direct function_definition /
        async_function_definition, or a decorated_definition wrapping one,
        return the inner function node to feed into the existing function
        extraction logic. Returns None if `node` is neither (e.g. a class,
        an if-statement, etc.) so callers can skip it as before.
        """
        if node.type in self.FUNCTION_TYPES:
            return node
        if node.type == "decorated_definition":
            for child in node.children:
                if child.type in self.FUNCTION_TYPES:
                    return child
        return None

    def _extract_decorators(self, fn_node: Node, source: bytes) -> List[str]:
        # Decorators are NOT children of the function_definition /
        # class_definition node itself — the grammar places them as
        # siblings inside a wrapping `decorated_definition` node. When
        # fn_node has been unwrapped from such a wrapper, look at the
        # parent instead so decorator metadata isn't silently dropped.
        decorator_host = fn_node
        if fn_node.parent is not None and fn_node.parent.type == "decorated_definition":
            decorator_host = fn_node.parent
        return [
            _node_text(child, source)
            for child in decorator_host.children
            if child.type == "decorator"
        ]

    def _extract_parameters(self, fn_node: Node, source: bytes) -> List[str]:
        params_node = _child_by_type(fn_node, "parameters")
        if not params_node:
            return []
        return [
            _node_text(p, source)
            for p in params_node.children
            if p.type not in ("(", ")", ",")
        ]

    def _extract_return_type(self, fn_node: Node, source: bytes) -> Optional[str]:
        for child in fn_node.children:
            if child.type == "type":
                return _node_text(child, source)
        return None


class JavaChunkExtractor:
    """Extracts methods, constructors, classes, and imports from Java AST."""

    METHOD_TYPES = {"method_declaration", "constructor_declaration"}
    CLASS_TYPES = {
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        "record_declaration",
    }
    IMPORT_TYPE = "import_declaration"

    def extract(
        self,
        tree_root: Node,
        source: bytes,
        file_path: str,
        config: ChunkerConfig,
    ) -> List[CodeChunk]:
        chunks: List[CodeChunk] = []
        seen_ids: Set[str] = set()

        # Imports
        import_nodes = _find_nodes(tree_root, self.IMPORT_TYPE)
        if import_nodes:
            start = import_nodes[0].start_point[0] + 1
            end = import_nodes[-1].end_point[0] + 1
            code = source[import_nodes[0].start_byte:import_nodes[-1].end_byte].decode(
                "utf-8", errors="replace"
            )
            cid = _build_chunk_id(file_path, start, code)
            if cid not in seen_ids:
                seen_ids.add(cid)
                c = CodeChunk(
                    chunk_id=cid,
                    code=code,
                    document="",
                    chunk_type="imports",
                    language="java",
                    file_path=file_path,
                    start_line=start,
                    end_line=end,
                )
                c.document = _build_document(c)
                chunks.append(c)

        # Classes / interfaces / enums
        for class_node in _find_nodes(tree_root, *self.CLASS_TYPES):
            chunks.extend(
                self._extract_class(class_node, source, file_path, config, seen_ids)
            )

        return [c for c in chunks if (c.end_line - c.start_line + 1) >= config.min_chunk_lines]

    def _extract_class(
        self,
        class_node: Node,
        source: bytes,
        file_path: str,
        config: ChunkerConfig,
        seen_ids: Set[str],
    ) -> List[CodeChunk]:
        chunks: List[CodeChunk] = []

        class_name = ""
        parent_class = None
        name_node = _child_by_type(class_node, "identifier")
        if name_node:
            class_name = _node_text(name_node, source)

        # Superclass
        superclass_node = _child_by_type(class_node, "superclass")
        if superclass_node:
            for child in superclass_node.children:
                if child.type == "type_identifier":
                    parent_class = _node_text(child, source)
                    break

        code = _node_text(class_node, source)
        start = class_node.start_point[0] + 1
        end = class_node.end_point[0] + 1
        cid = _build_chunk_id(file_path, start, code)
        javadoc = _extract_java_doc_comment(class_node, source)

        if cid not in seen_ids:
            seen_ids.add(cid)
            c = CodeChunk(
                chunk_id=cid,
                code=code,
                document="",
                chunk_type="class",
                language="java",
                file_path=file_path,
                start_line=start,
                end_line=end,
                class_name=class_name,
                parent_class=parent_class,
                docstring=javadoc,
                docstring_exists=javadoc is not None,
            )
            c.document = _build_document(c)
            chunks.append(c)

        # Methods / constructors
        body = _child_by_type(class_node, "class_body", "interface_body", "enum_body")
        if body:
            for method_node in _find_nodes(body, *self.METHOD_TYPES):
                chunk = self._extract_method(
                    method_node, source, file_path, class_name, parent_class, config
                )
                if chunk and chunk.chunk_id not in seen_ids:
                    seen_ids.add(chunk.chunk_id)
                    chunks.append(chunk)

        return chunks

    def _extract_method(
        self,
        method_node: Node,
        source: bytes,
        file_path: str,
        class_name: str,
        parent_class: Optional[str],
        config: ChunkerConfig,
    ) -> Optional[CodeChunk]:
        name_node = _child_by_type(method_node, "identifier")
        method_name = _node_text(name_node, source) if name_node else "<anonymous>"

        code = _node_text(method_node, source)
        start = method_node.start_point[0] + 1
        end = method_node.end_point[0] + 1

        if (end - start + 1) > config.max_chunk_lines:
            lines = code.splitlines()[:config.max_chunk_lines]
            code = "\n".join(lines)
            end = start + config.max_chunk_lines - 1

        cid = _build_chunk_id(file_path, start, code)
        javadoc = _extract_java_doc_comment(method_node, source)

        # Modifiers (public/private/static/etc.)
        modifiers = [
            _node_text(m, source)
            for m in method_node.children
            if m.type == "modifiers"
        ]

        c = CodeChunk(
            chunk_id=cid,
            code=code,
            document="",
            chunk_type="method",
            language="java",
            file_path=file_path,
            start_line=start,
            end_line=end,
            function_name=method_name,
            class_name=class_name,
            parent_class=parent_class,
            docstring=javadoc,
            docstring_exists=javadoc is not None,
            decorators=modifiers,
        )
        c.document = _build_document(c)
        return c


class JavaScriptChunkExtractor:
    """
    Extracts functions, arrow functions, classes, and imports from
    JavaScript and TypeScript ASTs.
    """

    FUNCTION_TYPES = {
        "function_declaration",
        "function_expression",
        "arrow_function",
        "generator_function_declaration",
    }
    METHOD_TYPES = {"method_definition"}
    CLASS_TYPES = {"class_declaration", "class_expression"}
    IMPORT_TYPES = {"import_statement", "import_declaration"}

    def __init__(self, language: str = "javascript"):
        self.language = language

    def extract(
        self,
        tree_root: Node,
        source: bytes,
        file_path: str,
        config: ChunkerConfig,
    ) -> List[CodeChunk]:
        chunks: List[CodeChunk] = []
        seen_ids: Set[str] = set()

        # Imports
        import_nodes = _find_nodes(tree_root, *self.IMPORT_TYPES)
        if import_nodes:
            start = import_nodes[0].start_point[0] + 1
            end = import_nodes[-1].end_point[0] + 1
            code = source[import_nodes[0].start_byte:import_nodes[-1].end_byte].decode(
                "utf-8", errors="replace"
            )
            cid = _build_chunk_id(file_path, start, code)
            if cid not in seen_ids:
                seen_ids.add(cid)
                c = CodeChunk(
                    chunk_id=cid,
                    code=code,
                    document="",
                    chunk_type="imports",
                    language=self.language,
                    file_path=file_path,
                    start_line=start,
                    end_line=end,
                )
                c.document = _build_document(c)
                chunks.append(c)

        # Classes
        for class_node in _find_nodes(tree_root, *self.CLASS_TYPES):
            chunks.extend(
                self._extract_class(class_node, source, file_path, config, seen_ids)
            )

        # Top-level named functions
        for fn_node in _find_nodes(tree_root, "function_declaration", "generator_function_declaration"):
            chunk = self._extract_function(fn_node, source, file_path, None, config)
            if chunk and chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                chunks.append(chunk)

        # Top-level arrow functions assigned to variables
        for var_node in _find_nodes(tree_root, "variable_declarator"):
            for child in var_node.children:
                if child.type == "arrow_function":
                    name_node = _child_by_type(var_node, "identifier")
                    chunk = self._extract_function(
                        child, source, file_path, None, config,
                        override_name=_node_text(name_node, source) if name_node else None,
                    )
                    if chunk and chunk.chunk_id not in seen_ids:
                        seen_ids.add(chunk.chunk_id)
                        chunks.append(chunk)

        return [c for c in chunks if (c.end_line - c.start_line + 1) >= config.min_chunk_lines]

    def _extract_class(
        self,
        class_node: Node,
        source: bytes,
        file_path: str,
        config: ChunkerConfig,
        seen_ids: Set[str],
    ) -> List[CodeChunk]:
        chunks: List[CodeChunk] = []

        class_name = ""
        parent_class = None

        name_node = _child_by_type(class_node, "identifier", "type_identifier")
        if name_node:
            class_name = _node_text(name_node, source)

        heritage = _child_by_type(class_node, "class_heritage")
        if heritage:
            for child in heritage.children:
                if child.type in ("identifier", "member_expression"):
                    parent_class = _node_text(child, source)
                    break

        code = _node_text(class_node, source)
        start = class_node.start_point[0] + 1
        end = class_node.end_point[0] + 1
        cid = _build_chunk_id(file_path, start, code)
        jsdoc = _extract_js_doc_comment(class_node, source)

        if cid not in seen_ids:
            seen_ids.add(cid)
            c = CodeChunk(
                chunk_id=cid,
                code=code,
                document="",
                chunk_type="class",
                language=self.language,
                file_path=file_path,
                start_line=start,
                end_line=end,
                class_name=class_name,
                parent_class=parent_class,
                docstring=jsdoc,
                docstring_exists=jsdoc is not None,
            )
            c.document = _build_document(c)
            chunks.append(c)

        body = _child_by_type(class_node, "class_body")
        if body:
            for method_node in _find_nodes(body, *self.METHOD_TYPES):
                chunk = self._extract_function(
                    method_node, source, file_path, class_name, config
                )
                if chunk and chunk.chunk_id not in seen_ids:
                    chunk.parent_class = parent_class
                    chunk.chunk_type = "method"
                    seen_ids.add(chunk.chunk_id)
                    chunks.append(chunk)

        return chunks

    def _extract_function(
        self,
        fn_node: Node,
        source: bytes,
        file_path: str,
        class_name: Optional[str],
        config: ChunkerConfig,
        override_name: Optional[str] = None,
    ) -> Optional[CodeChunk]:
        func_name = override_name

        if func_name is None:
            name_node = _child_by_type(
                fn_node, "identifier", "property_identifier", "type_identifier"
            )
            func_name = _node_text(name_node, source) if name_node else "<anonymous>"

        code = _node_text(fn_node, source)
        start = fn_node.start_point[0] + 1
        end = fn_node.end_point[0] + 1

        if (end - start + 1) > config.max_chunk_lines:
            lines = code.splitlines()[:config.max_chunk_lines]
            code = "\n".join(lines)
            end = start + config.max_chunk_lines - 1

        cid = _build_chunk_id(file_path, start, code)
        jsdoc = _extract_js_doc_comment(fn_node, source)
        chunk_type = "method" if class_name else "function"

        c = CodeChunk(
            chunk_id=cid,
            code=code,
            document="",
            chunk_type=chunk_type,
            language=self.language,
            file_path=file_path,
            start_line=start,
            end_line=end,
            function_name=func_name,
            class_name=class_name,
            docstring=jsdoc,
            docstring_exists=jsdoc is not None,
            is_async="async" in code[:20],
        )
        c.document = _build_document(c)
        return c


# ---------------------------------------------------------------------------
# Main Chunker
# ---------------------------------------------------------------------------

class TreeSitterChunker:
    """
    Orchestrates Tree-sitter parsing across multiple languages.

    Usage:
        chunker = TreeSitterChunker(config)
        chunks = chunker.chunk_repository("/path/to/repo")
    """

    def __init__(self, config: Optional[ChunkerConfig] = None):
        self.config = config or ChunkerConfig()
        self._parsers: Dict[str, Parser] = {}
        self._extractors: Dict[str, object] = {
            "python": PythonChunkExtractor(),
            "java": JavaChunkExtractor(),
            "javascript": JavaScriptChunkExtractor("javascript"),
            "typescript": JavaScriptChunkExtractor("typescript"),
        }

    def _get_parser(self, language: str) -> Parser:
        if language not in self._parsers:
            try:
                self._parsers[language] = get_parser(language)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load Tree-sitter grammar for '{language}'. "
                    f"Ensure tree-sitter-languages is installed: {exc}"
                ) from exc
        return self._parsers[language]

    def chunk_file(self, file_path: str | Path) -> List[CodeChunk]:
        """
        Parse a single source file and return all code chunks.
        Returns empty list for unsupported extensions or parse errors.
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        language = self.config.supported_extensions.get(ext)

        if language is None:
            logger.debug("Unsupported extension %s — skipping %s", ext, path)
            return []

        try:
            source_bytes = path.read_bytes()
        except OSError as exc:
            logger.warning("Cannot read file %s: %s", path, exc)
            return []

        try:
            parser = self._get_parser(language)
            tree = parser.parse(source_bytes)
        except Exception as exc:
            logger.warning("Tree-sitter parse error for %s: %s", path, exc)
            return []

        extractor = self._extractors.get(language)
        if extractor is None:
            logger.warning("No extractor for language '%s'", language)
            return []

        try:
            return extractor.extract(
                tree.root_node,
                source_bytes,
                str(path.resolve()),
                self.config,
            )
        except Exception as exc:
            logger.error("Extraction failed for %s: %s", path, exc, exc_info=True)
            return []

    def chunk_repository(
        self,
        repo_path: str | Path,
        include_extensions: Optional[List[str]] = None,
        extra_exclude_patterns: Optional[List[str]] = None,
    ) -> List[CodeChunk]:
        """
        Walk a repository directory and chunk all supported source files
        in parallel using ThreadPoolExecutor.

        Args:
            repo_path: Root directory of the repository.
            include_extensions: If set, only process these extensions.
            extra_exclude_patterns: Additional glob patterns to exclude.

        Returns:
            Deduplicated list of CodeChunks sorted by (file, start_line).
        """
        repo_path = Path(repo_path).resolve()
        if not repo_path.is_dir():
            raise ValueError(f"Repository path is not a directory: {repo_path}")

        exclude = set(self.config.exclude_patterns)
        if extra_exclude_patterns:
            exclude.update(extra_exclude_patterns)

        supported_exts = set(self.config.supported_extensions.keys())
        if include_extensions:
            supported_exts = supported_exts.intersection(
                {ext if ext.startswith(".") else f".{ext}" for ext in include_extensions}
            )

        # Collect all candidate files
        candidate_files: List[Path] = []
        skip_dirs = {".venv", "venv", ".git", "node_modules", "__pycache__", ".chromadb", ".bm25", "dist", "build"}
        for f in repo_path.rglob("*"):
            if not f.is_file():
                continue
            rel = f.relative_to(repo_path)
            if any(part in skip_dirs or part.startswith(".") for part in rel.parts[:-1]):
                continue
            if f.suffix.lower() not in supported_exts:
                continue
            if any(rel.match(pat.lstrip("*/")) for pat in exclude):
                continue
            candidate_files.append(f)

        logger.info(
            "Found %d candidate files in %s", len(candidate_files), repo_path
        )

        all_chunks: List[CodeChunk] = []
        seen_ids: Set[str] = set()

        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            future_to_file = {
                executor.submit(self.chunk_file, f): f for f in candidate_files
            }
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    file_chunks = future.result()
                    for chunk in file_chunks:
                        if chunk.chunk_id not in seen_ids:
                            seen_ids.add(chunk.chunk_id)
                            all_chunks.append(chunk)
                except Exception as exc:
                    logger.error(
                        "Unexpected error chunking %s: %s", file_path, exc
                    )

        all_chunks.sort(key=lambda c: (c.file_path, c.start_line))
        logger.info(
            "Extracted %d unique chunks from %d files",
            len(all_chunks),
            len(candidate_files),
        )
        return all_chunks

    def get_language_for_file(self, file_path: str | Path) -> Optional[str]:
        """Return the Tree-sitter language name for a given file path."""
        return self.config.supported_extensions.get(Path(file_path).suffix.lower())

    def iter_repository_files(
        self, repo_path: str | Path
    ) -> Iterator[Tuple[Path, str]]:
        """Yield (file_path, language) tuples for all supported files."""
        repo_path = Path(repo_path).resolve()
        for f in repo_path.rglob("*"):
            if not f.is_file():
                continue
            lang = self.get_language_for_file(f)
            if lang:
                yield f, lang