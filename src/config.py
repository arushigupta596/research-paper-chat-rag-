"""
Configuration management for the document understanding system.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Try to load Streamlit secrets if available (for Streamlit Cloud deployment)
try:
    import streamlit as st
    if hasattr(st, 'secrets'):
        # Override environment variables with Streamlit secrets
        for key in st.secrets:
            os.environ[key] = st.secrets[key]
except (ImportError, FileNotFoundError, KeyError):
    # Not running on Streamlit Cloud or secrets not configured
    pass

class Config(BaseModel):
    """System configuration."""

    # Paths
    PROJECT_ROOT: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "Data")
    PROCESSED_DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data" / "processed")
    CHROMA_PERSIST_DIR: Path = Field(
        default_factory=lambda: Path(os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
    )
    LOGS_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "logs")

    # API Configuration
    OPENROUTER_API_KEY: str = Field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    OPENROUTER_BASE_URL: str = Field(
        default_factory=lambda: os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    )

    # Model Configuration
    LLM_MODEL: str = Field(
        default_factory=lambda: os.getenv("LLM_MODEL", "openai/gpt-4-turbo-preview")
    )
    VLM_MODEL: str = Field(
        default_factory=lambda: os.getenv("VLM_MODEL", "qwen/qwen-vl-max")
    )
    EMBEDDING_MODEL: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
    )

    # Processing Configuration
    MAX_WORKERS: int = Field(default_factory=lambda: int(os.getenv("MAX_WORKERS", "4")))
    CHUNK_SIZE: int = Field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "512")))
    CHUNK_OVERLAP: int = Field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "50")))
    TOP_K_RETRIEVAL: int = Field(default_factory=lambda: int(os.getenv("TOP_K_RETRIEVAL", "10")))

    # OCR Configuration
    OCR_LANG: str = "en"
    OCR_USE_GPU: bool = True
    OCR_CONFIDENCE_THRESHOLD: float = 0.5

    # Layout Detection Configuration
    LAYOUT_MODEL: str = "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"
    LAYOUT_CONFIDENCE_THRESHOLD: float = 0.6

    # Region Types
    REGION_TYPES: list = ["text", "title", "list", "table", "figure"]

    def __init__(self, **data):
        super().__init__(**data)
        # Create directories if they don't exist
        self.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Global config instance
config = Config()
