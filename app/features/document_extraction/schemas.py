from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class DocumentStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"

class DocumentType(StrEnum):
    PDF = "pdf"
    JPG = "jpg"
    PNG = "png"
    TIFF = "tiff"
    XLSX = "xlsx"
    XLS = "xls"
    CSV = "csv"

class DocumentResponse(BaseModel):
    document_id: str = Field(min_length=1, description="Unique document id")
    filename: str = Field(min_length=1, description="Original file name of uploaded file")
    file_type: DocumentType
    file_size: int = Field(gt=0, description="Uploaded file size in bytes")
    status: DocumentStatus
    created_at: datetime
    total_page: int | None = Field(
        default=None,
        ge=1,
        description="Number of the pages for PDF documents",
    )

class UploadResponse(BaseModel):
    batch_id: str = Field(min_length=1, description="Unique identifier for the upload batch")
    total_files: int = Field(ge=1, description="Number of the uploaded documents")
    doucuments: list[DocumentResponse] = Field(
        min_length=1,
        description="Document register in the upload batch",
    )
    
