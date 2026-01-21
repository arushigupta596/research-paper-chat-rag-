"""
RAG retrieval layer with cross-paper synthesis and evidence linking.
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

from .vector_store import VectorStore
from ..config import config

logger = logging.getLogger(__name__)


@dataclass
class Evidence:
    """Container for evidence with source information."""
    text: str
    paper_name: str
    page_num: int
    region_type: str
    region_id: str
    bbox: List[float]
    score: float
    chunk_id: str


@dataclass
class RetrievalResult:
    """Container for retrieval results with grouped evidence."""
    query: str
    evidence_chunks: List[Evidence]
    papers_searched: List[str]
    total_chunks: int
    by_paper: Dict[str, List[Evidence]]
    by_region_type: Dict[str, List[Evidence]]


class RAGRetriever:
    """
    RAG retrieval layer with advanced features:
    - Cross-paper synthesis
    - Evidence grouping and ranking
    - Region-aware filtering
    - Diversity sampling
    """

    def __init__(self, vector_store: VectorStore):
        """
        Initialize RAG retriever.

        Args:
            vector_store: Initialized vector store
        """
        self.vector_store = vector_store
        logger.info("RAG retriever initialized")

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        filter_papers: Optional[List[str]] = None,
        filter_region_types: Optional[List[str]] = None,
        diversity_lambda: float = 0.5
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: User query
            top_k: Number of chunks to retrieve
            filter_papers: Optional list of papers to search
            filter_region_types: Optional list of region types to include
            diversity_lambda: Balance between relevance and diversity (0-1)

        Returns:
            RetrievalResult with organized evidence
        """
        top_k = top_k or config.TOP_K_RETRIEVAL

        logger.info(f"Retrieving for query: {query[:100]}...")

        # Build filter metadata
        filter_metadata = None
        if filter_papers and len(filter_papers) == 1:
            filter_metadata = {'paper_name': filter_papers[0]}
        elif filter_region_types and len(filter_region_types) == 1:
            filter_metadata = {'region_type': filter_region_types[0]}

        # Perform retrieval
        if filter_papers and len(filter_papers) > 1:
            # Multi-paper search
            results = self.vector_store.search_by_paper(
                query=query,
                paper_names=filter_papers,
                top_k=top_k
            )
        elif filter_region_types and len(filter_region_types) > 1:
            # Multi-region-type search
            results = self.vector_store.search_by_region_type(
                query=query,
                region_types=filter_region_types,
                top_k=top_k
            )
        else:
            # General search
            results = self.vector_store.search(
                query=query,
                top_k=top_k * 2,  # Get more for diversity sampling
                filter_metadata=filter_metadata
            )

        # Apply diversity sampling if needed
        if diversity_lambda > 0 and len(results) > top_k:
            results = self._diversify_results(results, top_k, diversity_lambda)
        else:
            results = results[:top_k]

        # Convert to Evidence objects
        evidence_chunks = []
        for result in results:
            metadata = result['metadata']
            evidence = Evidence(
                text=result['text'],
                paper_name=metadata['paper_name'],
                page_num=metadata['page_num'],
                region_type=metadata['region_type'],
                region_id=metadata['region_id'],
                bbox=eval(metadata['bbox']),  # Convert string back to list
                score=result['score'],
                chunk_id=result['chunk_id']
            )
            evidence_chunks.append(evidence)

        # Group evidence
        by_paper = self._group_by_paper(evidence_chunks)
        by_region_type = self._group_by_region_type(evidence_chunks)

        # Get searched papers
        papers_searched = list(by_paper.keys())

        retrieval_result = RetrievalResult(
            query=query,
            evidence_chunks=evidence_chunks,
            papers_searched=papers_searched,
            total_chunks=len(evidence_chunks),
            by_paper=by_paper,
            by_region_type=by_region_type
        )

        logger.info(
            f"Retrieved {len(evidence_chunks)} chunks from {len(papers_searched)} papers"
        )

        return retrieval_result

    def _diversify_results(
        self,
        results: List[Dict[str, Any]],
        top_k: int,
        diversity_lambda: float
    ) -> List[Dict[str, Any]]:
        """
        Apply maximal marginal relevance (MMR) for diversity.

        Args:
            results: Search results
            top_k: Number of results to select
            diversity_lambda: Balance between relevance and diversity

        Returns:
            Diversified results
        """
        if len(results) <= top_k:
            return results

        selected = [results[0]]  # Start with most relevant
        remaining = results[1:]

        while len(selected) < top_k and remaining:
            mmr_scores = []

            for candidate in remaining:
                # Relevance score
                relevance = candidate['score']

                # Diversity: minimum similarity to selected documents
                max_sim = max(
                    self._calculate_similarity(candidate, selected_doc)
                    for selected_doc in selected
                )

                # MMR score
                mmr = diversity_lambda * relevance - (1 - diversity_lambda) * max_sim
                mmr_scores.append((mmr, candidate))

            # Select best MMR score
            mmr_scores.sort(key=lambda x: x[0], reverse=True)
            best_candidate = mmr_scores[0][1]

            selected.append(best_candidate)
            remaining.remove(best_candidate)

        return selected

    def _calculate_similarity(
        self,
        doc1: Dict[str, Any],
        doc2: Dict[str, Any]
    ) -> float:
        """
        Calculate similarity between two documents.

        Simple heuristic based on paper name and region overlap.
        """
        # Same paper: higher similarity
        same_paper = doc1['metadata']['paper_name'] == doc2['metadata']['paper_name']

        # Same region: very high similarity
        same_region = doc1['metadata']['region_id'] == doc2['metadata']['region_id']

        if same_region:
            return 0.9
        elif same_paper:
            return 0.6
        else:
            return 0.3

    def _group_by_paper(
        self,
        evidence_chunks: List[Evidence]
    ) -> Dict[str, List[Evidence]]:
        """Group evidence by paper name."""
        grouped = defaultdict(list)
        for evidence in evidence_chunks:
            grouped[evidence.paper_name].append(evidence)
        return dict(grouped)

    def _group_by_region_type(
        self,
        evidence_chunks: List[Evidence]
    ) -> Dict[str, List[Evidence]]:
        """Group evidence by region type."""
        grouped = defaultdict(list)
        for evidence in evidence_chunks:
            grouped[evidence.region_type].append(evidence)
        return dict(grouped)

    def get_context_for_llm(
        self,
        retrieval_result: RetrievalResult,
        max_tokens: int = 4000
    ) -> str:
        """
        Format retrieval results as context for LLM.

        Args:
            retrieval_result: Retrieval result
            max_tokens: Maximum tokens for context

        Returns:
            Formatted context string
        """
        context_parts = []
        context_parts.append(f"Query: {retrieval_result.query}\n")
        context_parts.append(
            f"Retrieved {retrieval_result.total_chunks} relevant passages "
            f"from {len(retrieval_result.papers_searched)} papers.\n\n"
        )

        # Estimate tokens (rough: 4 chars per token)
        current_tokens = len(''.join(context_parts)) // 4

        for i, evidence in enumerate(retrieval_result.evidence_chunks, 1):
            # Format evidence
            evidence_text = (
                f"[Evidence {i}]\n"
                f"Source: {evidence.paper_name}, Page {evidence.page_num}\n"
                f"Type: {evidence.region_type}\n"
                f"Relevance: {evidence.score:.3f}\n"
                f"Content:\n{evidence.text}\n\n"
            )

            evidence_tokens = len(evidence_text) // 4

            if current_tokens + evidence_tokens > max_tokens:
                context_parts.append(
                    f"\n[Note: Showing {i-1} of {retrieval_result.total_chunks} "
                    "retrieved passages due to context limit]\n"
                )
                break

            context_parts.append(evidence_text)
            current_tokens += evidence_tokens

        return ''.join(context_parts)

    def format_evidence_for_display(
        self,
        evidence: Evidence
    ) -> Dict[str, Any]:
        """
        Format evidence for display in UI.

        Args:
            evidence: Evidence object

        Returns:
            Dictionary with formatted evidence
        """
        return {
            'text': evidence.text,
            'source': {
                'paper': evidence.paper_name,
                'page': evidence.page_num,
                'region_type': evidence.region_type,
                'region_id': evidence.region_id
            },
            'score': round(evidence.score, 3),
            'citation': f"{evidence.paper_name}, Page {evidence.page_num}"
        }
