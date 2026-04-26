from fastapi import APIRouter

from src.api.schemas import HealthResponse
from src.version import __version__

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(version=__version__)
