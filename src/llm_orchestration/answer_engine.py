"""
Answer synthesis engine using LangChain and OpenRouter.
"""
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..retrieval.rag_retriever import RAGRetriever, RetrievalResult, Evidence
from ..config import config

logger = logging.getLogger(__name__)


@dataclass
class Answer:
    """Container for answer with evidence."""
    question: str
    answer: str
    evidence: List[Dict[str, Any]]
    sources: List[str]
    has_evidence: bool
    retrieval_stats: Dict[str, Any]


class AnswerEngine:
    """
    Answer synthesis engine with evidence backing.

    Uses LangChain + OpenRouter for:
    - Evidence-based answering
    - Citation generation
    - Multi-document reasoning
    """

    def __init__(
        self,
        retriever: RAGRetriever,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize answer engine.

        Args:
            retriever: RAG retriever
            api_key: OpenRouter API key
            model: Model name
        """
        self.retriever = retriever
        self.api_key = api_key or config.OPENROUTER_API_KEY
        self.model = model or config.LLM_MODEL

        # Load paper metadata
        self.paper_metadata = self._load_paper_metadata()

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.model,
            openai_api_key=self.api_key,
            openai_api_base=config.OPENROUTER_BASE_URL,
            temperature=0.1,  # Low temperature for factual accuracy
            max_tokens=1200  # Reduced to avoid credit errors
        )

        # System prompt for evidence-based answering
        self.system_prompt = """You are a research assistant specialized in analyzing academic papers.

Your task is to answer questions based ONLY on the provided evidence from research papers. Follow these rules strictly:

1. ONLY use information explicitly stated in the provided evidence passages
2. NEVER use external knowledge or make assumptions
3. Write your answer in a natural, conversational style directly addressing the question
4. When referencing information, cite the evidence inline using [Evidence N] notation
5. When mentioning findings, include the paper's topic/subject area naturally in your text (e.g., "According to the paper on Text-to-Speech Generation..." or "The Multi-Modal Learning study shows...")
6. Be precise and accurate - do not paraphrase in ways that change meaning
7. Synthesize information across papers when relevant, clearly indicating which paper (with its topic) each insight comes from
8. If evidence passages contradict each other, acknowledge the contradiction and present both viewpoints
9. If the evidence doesn't contain enough information to answer, explicitly state: "The provided evidence does not contain sufficient information to answer this question."

IMPORTANT - Format your response in THREE sections:

First, write your complete answer naturally with inline citations [Evidence N]. Make it conversational and comprehensive.

Then, after your answer is complete, add:

---
**Evidence:**
- Evidence 1 (from [Paper Topic]): [Brief description of what this evidence specifically shows]
- Evidence 2 (from [Paper Topic]): [Brief description]
[Continue for each evidence cited]

**Sources:**
- [Paper Title] (Topic: [Topic]) - Pages referenced
[List all unique papers used]
"""

        logger.info(f"Answer engine initialized with model: {self.model}")

    def _load_paper_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load paper metadata from JSON file."""
        metadata_path = config.DATA_DIR / 'paper_metadata.json'
        try:
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            else:
                logger.warning("Paper metadata file not found, using filenames only")
                return {}
        except Exception as e:
            logger.error(f"Failed to load paper metadata: {e}")
            return {}

    def answer_question(
        self,
        question: str,
        top_k: int = None,
        filter_papers: Optional[List[str]] = None,
        filter_region_types: Optional[List[str]] = None
    ) -> Answer:
        """
        Answer a question with evidence backing.

        Args:
            question: User question
            top_k: Number of evidence chunks to retrieve
            filter_papers: Optional paper filter
            filter_region_types: Optional region type filter

        Returns:
            Answer object with evidence and sources
        """
        logger.info(f"Answering question: {question[:100]}...")

        # Step 1: Retrieve relevant evidence
        retrieval_result = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            filter_papers=filter_papers,
            filter_region_types=filter_region_types
        )

        if not retrieval_result.evidence_chunks:
            return Answer(
                question=question,
                answer="No relevant evidence found in the indexed papers.",
                evidence=[],
                sources=[],
                has_evidence=False,
                retrieval_stats={
                    'total_chunks': 0,
                    'papers_searched': []
                }
            )

        # Step 2: Format context for LLM
        context = self._format_context(retrieval_result)

        # Step 3: Generate answer
        answer_text = self._generate_answer(question, context)

        # Step 4: Format evidence for response
        evidence_list = [
            self.retriever.format_evidence_for_display(ev)
            for ev in retrieval_result.evidence_chunks
        ]

        # Step 5: Extract unique sources
        sources = self._extract_sources(retrieval_result.evidence_chunks)

        # Create answer object
        answer = Answer(
            question=question,
            answer=answer_text,
            evidence=evidence_list,
            sources=sources,
            has_evidence=True,
            retrieval_stats={
                'total_chunks': retrieval_result.total_chunks,
                'papers_searched': retrieval_result.papers_searched,
                'by_paper': {
                    paper: len(chunks)
                    for paper, chunks in retrieval_result.by_paper.items()
                },
                'by_region_type': {
                    rtype: len(chunks)
                    for rtype, chunks in retrieval_result.by_region_type.items()
                }
            }
        )

        logger.info("Answer generated successfully")
        return answer

    def _format_context(self, retrieval_result: RetrievalResult) -> str:
        """
        Format retrieval results as context for LLM.

        Args:
            retrieval_result: Retrieval result

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, evidence in enumerate(retrieval_result.evidence_chunks, 1):
            # Get paper metadata
            paper_meta = self.paper_metadata.get(evidence.paper_name, {})
            paper_topic = paper_meta.get('topic', 'Unknown Topic')
            paper_title = paper_meta.get('title', evidence.paper_name)

            context_parts.append(
                f"[Evidence {i}]\n"
                f"Paper: {paper_title}\n"
                f"Topic: {paper_topic}\n"
                f"Source: {evidence.paper_name}, Page {evidence.page_num}\n"
                f"Region Type: {evidence.region_type}\n"
                f"Content:\n{evidence.text}\n"
            )

        return "\n".join(context_parts)

    def _generate_answer(self, question: str, context: str) -> str:
        """
        Generate answer using LLM.

        Args:
            question: User question
            context: Formatted evidence context

        Returns:
            Generated answer
        """
        # Create prompt
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(
                content=f"Evidence passages:\n\n{context}\n\n"
                        f"Question: {question}\n\n"
                        "Please provide an evidence-backed answer following the format specified."
            )
        ]

        try:
            # Generate answer
            response = self.llm.invoke(messages)
            answer = response.content

            return answer

        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return f"Error generating answer: {str(e)}"

    def _extract_sources(self, evidence_chunks: List[Evidence]) -> List[str]:
        """
        Extract unique source citations from evidence with paper topics.

        Args:
            evidence_chunks: List of evidence chunks

        Returns:
            List of unique source citations with topics
        """
        # Get unique papers
        papers = {}
        for evidence in evidence_chunks:
            if evidence.paper_name not in papers:
                paper_meta = self.paper_metadata.get(evidence.paper_name, {})
                paper_title = paper_meta.get('title', evidence.paper_name)
                paper_topic = paper_meta.get('topic', 'General Research')
                papers[evidence.paper_name] = {
                    'title': paper_title,
                    'topic': paper_topic,
                    'pages': set()
                }
            papers[evidence.paper_name]['pages'].add(evidence.page_num)

        # Format sources with topics
        sources = []
        for paper_name, info in sorted(papers.items()):
            pages = sorted(list(info['pages']))
            page_str = f"Pages {', '.join(map(str, pages))}" if pages else "N/A"
            sources.append(f"{info['title']} (Topic: {info['topic']}) - {page_str}")

        return sources

    def multi_hop_reasoning(
        self,
        question: str,
        max_hops: int = 3
    ) -> Answer:
        """
        Perform multi-hop reasoning for complex questions.

        This is useful for questions that require synthesizing information
        across multiple pieces of evidence or reasoning chains.

        Args:
            question: Complex question requiring multi-hop reasoning
            max_hops: Maximum number of reasoning hops

        Returns:
            Answer with multi-hop reasoning
        """
        logger.info(f"Performing multi-hop reasoning for: {question[:100]}...")

        # Initial retrieval
        current_query = question
        all_evidence = []
        reasoning_chain = []

        for hop in range(max_hops):
            logger.debug(f"Reasoning hop {hop + 1}/{max_hops}")

            # Retrieve for current query
            retrieval_result = self.retriever.retrieve(
                query=current_query,
                top_k=5
            )

            if not retrieval_result.evidence_chunks:
                break

            all_evidence.extend(retrieval_result.evidence_chunks)

            # Generate intermediate reasoning
            context = self._format_context(retrieval_result)
            intermediate_answer = self._generate_answer(current_query, context)

            reasoning_chain.append({
                'hop': hop + 1,
                'query': current_query,
                'answer': intermediate_answer
            })

            # Check if we have enough information
            if "sufficient information" in intermediate_answer.lower():
                break

            # Generate follow-up query if needed
            if hop < max_hops - 1:
                current_query = self._generate_followup_query(
                    question,
                    intermediate_answer
                )

        # Final synthesis
        final_context = self._format_context(
            RetrievalResult(
                query=question,
                evidence_chunks=all_evidence,
                papers_searched=list(set(ev.paper_name for ev in all_evidence)),
                total_chunks=len(all_evidence),
                by_paper={},
                by_region_type={}
            )
        )

        final_answer = self._generate_answer(question, final_context)

        # Format evidence
        evidence_list = [
            self.retriever.format_evidence_for_display(ev)
            for ev in all_evidence
        ]

        sources = self._extract_sources(all_evidence)

        return Answer(
            question=question,
            answer=f"Multi-hop Reasoning ({len(reasoning_chain)} hops):\n\n{final_answer}",
            evidence=evidence_list,
            sources=sources,
            has_evidence=True,
            retrieval_stats={
                'total_chunks': len(all_evidence),
                'reasoning_hops': len(reasoning_chain),
                'papers_searched': list(set(ev.paper_name for ev in all_evidence))
            }
        )

    def _generate_followup_query(
        self,
        original_question: str,
        intermediate_answer: str
    ) -> str:
        """
        Generate a follow-up query for multi-hop reasoning.

        Args:
            original_question: Original question
            intermediate_answer: Previous hop's answer

        Returns:
            Follow-up query
        """
        prompt = f"""Original question: {original_question}

Previous finding: {intermediate_answer}

What additional information is needed to fully answer the original question?
Generate a focused follow-up query (one sentence).
"""

        messages = [HumanMessage(content=prompt)]

        try:
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate follow-up query: {e}")
            return original_question
