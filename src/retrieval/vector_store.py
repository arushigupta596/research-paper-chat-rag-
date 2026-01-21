"""
ChromaDB vector store for semantic search and retrieval.
"""
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from ..config import config

logger = logging.getLogger(__name__)


class VectorStore:
    """
    ChromaDB-based vector store for document chunks.

    Supports:
    - Multi-document indexing
    - Metadata filtering
    - Cross-paper retrieval
    - Region-aware search
    """

    def __init__(
        self,
        collection_name: str = "research_papers",
        persist_directory: Optional[Path] = None,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_model: Name of the embedding model
        """
        self.collection_name = collection_name
        self.persist_dir = persist_directory or config.CHROMA_PERSIST_DIR
        self.embedding_model_name = embedding_model or config.EMBEDDING_MODEL

        # Initialize embedding model
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(f"Vector store initialized: {self.collection_name}")
        logger.info(f"Current collection size: {self.collection.count()}")

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Add chunks to the vector store.

        Args:
            chunks: List of chunk dictionaries with text and metadata
        """
        if not chunks:
            logger.warning("No chunks to add")
            return

        logger.info(f"Adding {len(chunks)} chunks to vector store")

        # Prepare data for ChromaDB
        ids = []
        texts = []
        metadatas = []

        for chunk in chunks:
            ids.append(chunk['chunk_id'])
            texts.append(chunk['text'])

            # Prepare metadata (ChromaDB requires flat dictionary)
            metadata = {
                'paper_name': chunk['paper_name'],
                'page_num': chunk['page_num'],
                'region_id': chunk['region_id'],
                'region_type': chunk['region_type'],
                'reading_order': chunk['reading_order'],
                'chunk_index': chunk['chunk_index'],
                'bbox': str(chunk['bbox'])  # Store as string
            }

            if chunk.get('section'):
                metadata['section'] = chunk['section']

            metadatas.append(metadata)

        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        ).tolist()

        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        logger.info(f"Successfully added {len(chunks)} chunks")
        logger.info(f"Total collection size: {self.collection.count()}")

    def search(
        self,
        query: str,
        top_k: int = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks.

        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of relevant chunks with scores
        """
        top_k = top_k or config.TOP_K_RETRIEVAL

        logger.debug(f"Searching for: {query[:100]}...")

        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True
        ).tolist()

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata
        )

        # Format results
        formatted_results = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                result = {
                    'chunk_id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'score': 1 - results['distances'][0][i]  # Convert distance to similarity
                }
                formatted_results.append(result)

        logger.debug(f"Found {len(formatted_results)} results")
        return formatted_results

    def search_by_paper(
        self,
        query: str,
        paper_names: List[str],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Search within specific papers.

        Args:
            query: Search query
            paper_names: List of paper names to search within
            top_k: Number of results per paper

        Returns:
            List of relevant chunks
        """
        results = []

        for paper_name in paper_names:
            paper_results = self.search(
                query=query,
                top_k=top_k,
                filter_metadata={'paper_name': paper_name}
            )
            results.extend(paper_results)

        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)

        return results[:top_k] if top_k else results

    def search_by_region_type(
        self,
        query: str,
        region_types: List[str],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Search within specific region types (e.g., tables, figures).

        Args:
            query: Search query
            region_types: List of region types to search
            top_k: Number of results

        Returns:
            List of relevant chunks
        """
        results = []

        for region_type in region_types:
            type_results = self.search(
                query=query,
                top_k=top_k,
                filter_metadata={'region_type': region_type}
            )
            results.extend(type_results)

        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)

        return results[:top_k] if top_k else results

    def get_all_papers(self) -> List[str]:
        """
        Get list of all indexed paper names.

        Returns:
            List of unique paper names
        """
        # Get all documents
        all_docs = self.collection.get()

        # Extract unique paper names
        paper_names = set()
        if all_docs['metadatas']:
            for metadata in all_docs['metadatas']:
                paper_names.add(metadata['paper_name'])

        return sorted(list(paper_names))

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with statistics
        """
        all_docs = self.collection.get()

        stats = {
            'total_chunks': self.collection.count(),
            'papers': [],
            'region_type_counts': {}
        }

        if all_docs['metadatas']:
            # Count by paper
            paper_counts = {}
            region_counts = {}

            for metadata in all_docs['metadatas']:
                paper_name = metadata['paper_name']
                region_type = metadata['region_type']

                paper_counts[paper_name] = paper_counts.get(paper_name, 0) + 1
                region_counts[region_type] = region_counts.get(region_type, 0) + 1

            stats['papers'] = [
                {'name': name, 'chunks': count}
                for name, count in sorted(paper_counts.items())
            ]
            stats['region_type_counts'] = region_counts

        return stats

    def delete_paper(self, paper_name: str):
        """
        Delete all chunks from a specific paper.

        Args:
            paper_name: Name of the paper to delete
        """
        # Get all chunk IDs for this paper
        results = self.collection.get(
            where={'paper_name': paper_name}
        )

        if results['ids']:
            self.collection.delete(ids=results['ids'])
            logger.info(f"Deleted {len(results['ids'])} chunks from {paper_name}")
        else:
            logger.warning(f"No chunks found for paper: {paper_name}")

    def clear(self):
        """Clear all data from the collection."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Cleared vector store")
