from fastapi import APIRouter

from app.api.v1.health import router as health_router
from app.features.document_extraction.router import router as document_router
from app.features.invoices.router import router as invoice_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(document_router)
api_router.include_router(invoice_router)
