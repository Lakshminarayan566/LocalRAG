"""
ChromaDB vector store for semantic code search
Provides efficient retrieval with cosine similarity
"""

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import time

from tree_sitter_chunker import CodeChunk
from config import ChromaDBConfig


class CodeVectorStore:
    """
    Vector store for code chunks using ChromaDB
    Enables semantic search with configurable similarity metrics
    """
    
    def __init__(self, config: ChromaDBConfig):
        self.config = config
        self.client = None
        self.collection = None
        self.embedding_function = None
        self._initialize()
    
    def _initialize(self):
        """Initialize ChromaDB client and collection"""
        # Create persistent client
        self.client = chromadb.PersistentClient(
            path=self.config.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Setup embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.config.embedding_model
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=self.config.collection_name,
                embedding_function=self.embedding_function
            )
            print(f"Loaded existing collection: {self.config.collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=self.config.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": self.config.distance_metric}
            )
            print(f"Created new collection: {self.config.collection_name}")
    
    def add_chunks(self, chunks: List[CodeChunk], batch_size: int = 100) -> int:
        """
        Add code chunks to vector store
        
        Args:
            chunks: List of CodeChunk objects
            batch_size: Batch size for insertion
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0
        
        total_added = 0
        
        # Process in batches for efficiency
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Prepare data for ChromaDB
            ids = [chunk.chunk_id for chunk in batch]
            documents = [chunk.content for chunk in batch]
            metadatas = [
                {
                    'chunk_type': chunk.chunk_type,
                    'language': chunk.language,
                    'file_path': chunk.file_path,
                    'start_line': chunk.start_line,
                    'end_line': chunk.end_line,
                    'name': chunk.name or '',
                    'parent_context': chunk.parent_context or '',
                    **chunk.metadata
                }
                for chunk in batch
            ]
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
            total_added += len(batch)
            print(f"Added batch {i//batch_size + 1}: {len(batch)} chunks")
        
        print(f"Total chunks added: {total_added}")
        return total_added
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Semantic search for relevant code chunks
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_dict: Metadata filters (e.g., {'language': 'python'})
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of search results with chunks and scores
        """
        if top_k is None:
            top_k = self.config.top_k
        if similarity_threshold is None:
            similarity_threshold = self.config.similarity_threshold
        
        # Perform search
        start_time = time.time()
        
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k * 2,  # Get more to filter by threshold
            where=filter_dict
        )
        
        search_time = time.time() - start_time
        
        # Process results
        retrieved_chunks = []
        
        if results['ids'][0]:
            for i, chunk_id in enumerate(results['ids'][0]):
                # Convert distance to similarity score (for cosine distance)
                distance = results['distances'][0][i]
                similarity = 1 - distance  # ChromaDB returns cosine distance
                
                # Apply threshold
                if similarity >= similarity_threshold:
                    retrieved_chunks.append({
                        'chunk_id': chunk_id,
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'similarity': similarity,
                        'rank': i + 1
                    })
        
        # Limit to top_k after filtering
        retrieved_chunks = retrieved_chunks[:top_k]
        
        print(f"Search completed in {search_time:.3f}s, found {len(retrieved_chunks)} results")
        
        return retrieved_chunks
    
    def search_by_file(self, file_path: str, query: str, top_k: int = 5) -> List[Dict]:
        """Search within a specific file"""
        return self.search(
            query=query,
            top_k=top_k,
            filter_dict={'file_path': file_path}
        )
    
    def search_by_language(self, language: str, query: str, top_k: int = 5) -> List[Dict]:
        """Search within a specific programming language"""
        return self.search(
            query=query,
            top_k=top_k,
            filter_dict={'language': language}
        )
    
    def search_functions(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search only functions"""
        return self.search(
            query=query,
            top_k=top_k,
            filter_dict={'chunk_type': 'function'}
        )
    
    def search_classes(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search only classes"""
        return self.search(
            query=query,
            top_k=top_k,
            filter_dict={'chunk_type': 'class'}
        )
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """Retrieve a specific chunk by ID"""
        result = self.collection.get(
            ids=[chunk_id],
            include=['documents', 'metadatas']
        )
        
        if result['ids']:
            return {
                'chunk_id': result['ids'][0],
                'content': result['documents'][0],
                'metadata': result['metadatas'][0]
            }
        return None
    
    def delete_by_file(self, file_path: str) -> int:
        """Delete all chunks from a specific file"""
        # Get all chunks for the file
        results = self.collection.get(
            where={'file_path': file_path}
        )
        
        if results['ids']:
            self.collection.delete(ids=results['ids'])
            return len(results['ids'])
        return 0
    
    def update_chunk(self, chunk: CodeChunk):
        """Update an existing chunk"""
        # Delete old version
        self.collection.delete(ids=[chunk.chunk_id])
        
        # Add new version
        self.add_chunks([chunk])
    
    def get_statistics(self) -> Dict:
        """Get collection statistics"""
        count = self.collection.count()
        
        # Get sample to analyze
        sample = self.collection.get(limit=100)
        
        stats = {
            'total_chunks': count,
            'collection_name': self.config.collection_name,
            'embedding_model': self.config.embedding_model,
        }
        
        if sample['metadatas']:
            # Count by type
            chunk_types = {}
            languages = {}
            
            for metadata in sample['metadatas']:
                chunk_type = metadata.get('chunk_type', 'unknown')
                language = metadata.get('language', 'unknown')
                
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                languages[language] = languages.get(language, 0) + 1
            
            stats['chunk_types'] = chunk_types
            stats['languages'] = languages
        
        return stats
    
    def reset_collection(self):
        """Delete all data in collection"""
        self.client.delete_collection(self.config.collection_name)
        self._initialize()
        print("Collection reset completed")
    
    def export_chunks(self, output_path: str):
        """Export all chunks to file"""
        import json
        
        # Get all chunks
        results = self.collection.get(
            include=['documents', 'metadatas']
        )
        
        chunks_data = []
        for i, chunk_id in enumerate(results['ids']):
            chunks_data.append({
                'id': chunk_id,
                'content': results['documents'][i],
                'metadata': results['metadatas'][i]
            })
        
        with open(output_path, 'w') as f:
            json.dump(chunks_data, f, indent=2)
        
        print(f"Exported {len(chunks_data)} chunks to {output_path}")
    
    def import_chunks(self, input_path: str):
        """Import chunks from file"""
        import json
        
        with open(input_path, 'r') as f:
            chunks_data = json.load(f)
        
        # Convert to CodeChunk objects
        chunks = []
        for chunk_data in chunks_data:
            metadata = chunk_data['metadata']
            chunk = CodeChunk(
                content=chunk_data['content'],
                chunk_type=metadata['chunk_type'],
                language=metadata['language'],
                file_path=metadata['file_path'],
                start_line=metadata['start_line'],
                end_line=metadata['end_line'],
                name=metadata.get('name'),
                parent_context=metadata.get('parent_context'),
                chunk_id=chunk_data['id']
            )
            chunks.append(chunk)
        
        self.add_chunks(chunks)
        print(f"Imported {len(chunks)} chunks from {input_path}")
