"""
Data Contracts for the LégiRoute RAG system.

These Pydantic models enforce type safety across the entire pipeline:
Ingestion → Indexing → Retrieval → Generation.

"""

from pydantic import BaseModel, Field, field_validator, computed_field
from typing import Optional


class TrafficLawArticle(BaseModel):
    """
    Represents a single article from the French Traffic Code (Code de la Route).
    Acts as a strict contract: data must adhere to this schema or be rejected.
    """
    id: str = Field(..., description="Unique LEGI identifier (e.g., LEGIARTI0000...).")
    article_number: str = Field(..., description="Article number (e.g., R413-17).")
    content: str = Field(..., description="Raw text content of the article.")
    context: str = Field(..., description="Hierarchical path (Code > Book > Title...).")
    
    # Optional metadata for future-proofing
    url: Optional[str] = None 
    
    @field_validator('content')
    @classmethod
    def validate_content_length(cls, v: str) -> str:
        """Validates that the article content is not empty or too short."""
        if not v or len(v.strip()) < 5:
            raise ValueError("Article content is empty or too short.")
        return v

    @computed_field
    def blob_for_embedding(self) -> str:
        """
        Constructs the text blob used for vector embedding.
        Encapsulates business logic to ensure the model 'sees' the full hierarchy.
        """
        return f"{self.context} \nArticle {self.article_number} : {self.content}"

    @computed_field
    def full_url(self) -> str:
        """Reconstructs the official Légifrance URL."""
        return f"https://www.legifrance.gouv.fr/codes/article_lc/{self.id}"


class RetrievalResult(BaseModel):
    """
    Standardized output for the retrieval system.
    """
    article: TrafficLawArticle
    score: float = Field(..., description="Similarity score.")
    
    def __str__(self):
        return f"[{self.score:.4f}] {self.article.article_number}"
