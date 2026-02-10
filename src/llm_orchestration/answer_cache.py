"""
Answer caching system for suggested questions.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CachedAnswer:
    """Cached answer with metadata."""
    question: str
    answer: str
    evidence: list
    sources: list
    has_evidence: bool
    retrieval_stats: dict
    cached_at: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CachedAnswer':
        """Create from dictionary."""
        return cls(**data)


class AnswerCache:
    """
    Cache manager for pre-computed answers to suggested questions.

    This allows the app to instantly display answers for common questions
    without calling the LLM, improving response time and reducing API costs.
    """

    def __init__(self, cache_file: Path):
        """
        Initialize cache manager.

        Args:
            cache_file: Path to cache JSON file
        """
        self.cache_file = cache_file
        self.cache: Dict[str, CachedAnswer] = {}
        self._load_cache()

    def _load_cache(self):
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.cache = {
                        q: CachedAnswer.from_dict(a)
                        for q, a in data.items()
                    }
                logger.info(f"Loaded {len(self.cache)} cached answers")
            except Exception as e:
                logger.error(f"Failed to load cache: {e}")
                self.cache = {}
        else:
            logger.info("No cache file found, starting fresh")
            self.cache = {}

    def _save_cache(self):
        """Save cache to disk."""
        try:
            # Ensure directory exists
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)

            # Convert to JSON-serializable format
            data = {
                q: a.to_dict()
                for q, a in self.cache.items()
            }

            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {len(self.cache)} cached answers")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def get(self, question: str) -> Optional[CachedAnswer]:
        """
        Get cached answer for a question.

        Args:
            question: Question text

        Returns:
            Cached answer if exists and valid, None otherwise
        """
        cached_answer = self.cache.get(question)

        # Check if answer contains errors
        if cached_answer and cached_answer.answer:
            if 'Error generating answer' in cached_answer.answer or 'Error code: 402' in cached_answer.answer:
                logger.warning(f"Cached answer for '{question[:50]}...' contains error, will regenerate")
                return None

        return cached_answer

    def has(self, question: str) -> bool:
        """
        Check if question is cached with a valid answer.

        Args:
            question: Question text

        Returns:
            True if cached and valid, False otherwise
        """
        if question not in self.cache:
            return False

        # Check if cached answer is valid (not an error message)
        cached_answer = self.cache[question]
        if 'Error generating answer' in cached_answer.answer or 'Error code: 402' in cached_answer.answer:
            return False

        return True

    def set(self, question: str, answer_obj: Any):
        """
        Cache an answer.

        Args:
            question: Question text
            answer_obj: Answer object from AnswerEngine
        """
        cached_answer = CachedAnswer(
            question=answer_obj.question,
            answer=answer_obj.answer,
            evidence=answer_obj.evidence,
            sources=answer_obj.sources,
            has_evidence=answer_obj.has_evidence,
            retrieval_stats=answer_obj.retrieval_stats,
            cached_at=datetime.now().isoformat()
        )

        self.cache[question] = cached_answer
        self._save_cache()
        logger.info(f"Cached answer for: {question[:50]}...")

    def update_all(self, questions: list, answer_engine: Any, **kwargs):
        """
        Update cache for all suggested questions.

        This should be run periodically to refresh cached answers,
        especially after document updates.

        Args:
            questions: List of questions to cache
            answer_engine: AnswerEngine instance
            **kwargs: Additional arguments for answer_question
        """
        logger.info(f"Updating cache for {len(questions)} questions...")

        for i, question in enumerate(questions, 1):
            logger.info(f"Processing {i}/{len(questions)}: {question[:50]}...")

            try:
                # Generate answer
                answer = answer_engine.answer_question(
                    question=question,
                    **kwargs
                )

                # Cache it
                self.set(question, answer)

            except Exception as e:
                logger.error(f"Failed to cache question '{question}': {e}")

        logger.info("Cache update complete")

    def clear(self):
        """Clear all cached answers."""
        self.cache = {}
        self._save_cache()
        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'total_cached': len(self.cache),
            'questions': list(self.cache.keys())
        }
