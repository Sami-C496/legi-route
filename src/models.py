from pydantic import BaseModel, field_validator, computed_field


class TrafficLawArticle(BaseModel):
    id: str
    article_number: str
    content: str
    context: str

    @field_validator("content")
    @classmethod
    def validate_content_length(cls, v: str) -> str:
        if not v or len(v.strip()) < 5:
            raise ValueError("Article content is empty or too short.")
        return v

    @computed_field
    def blob_for_embedding(self) -> str:
        return f"{self.context}\nArticle {self.article_number} : {self.content}"

    @computed_field
    def full_url(self) -> str:
        return f"https://www.legifrance.gouv.fr/codes/article_lc/{self.id}"


class RetrievalResult(BaseModel):
    article: TrafficLawArticle
    score: float

    def __str__(self):
        return f"[{self.score:.4f}] {self.article.article_number}"
