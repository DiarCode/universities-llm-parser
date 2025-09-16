# domain/dto.py
from pydantic import BaseModel, Field
from typing import List, Optional
from domain.enums import (
    INSTITUTION_FINANCING_TYPE,
    INSTITUTION_TYPE,
    ENROLLMENT_REQUIREMENT_TYPE,
    MAJOR_CATEGORY,
)

class CountryRow(BaseModel):
    id: int
    name: str
    emoji: Optional[str] = None

class CityRow(BaseModel):
    id: int
    countryId: int
    name: str

class CreateInstitutionEnrollmentDocumentDto(BaseModel):
    name: str

class CreateInstitutionEnrollmentRequirementDto(BaseModel):
    name: str
    type: ENROLLMENT_REQUIREMENT_TYPE
    value: Optional[str] = None

class CreateInstitutionMajorDto(BaseModel):
    name: str
    durationYears: Optional[int] = None
    learningLanguage: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None
    category: MAJOR_CATEGORY
    requirements: Optional[List[str]] = None

class CreateInstitutionBulkDto(BaseModel):
    # Все текстовые поля (кроме enum) — на русском
    name: str
    shortName: Optional[str] = Field(default=None, alias="shortName")
    description: Optional[str] = None
    foundationYear: Optional[str] = None
    financingType: INSTITUTION_FINANCING_TYPE
    type: INSTITUTION_TYPE
    website: Optional[str] = None
    email: Optional[str] = None
    contactNumber: Optional[str] = None

    # ВНИМАНИЕ: теперь строка — передаём НАЗВАНИЕ города
    cityId: str

    # Backend на загрузке найдет city по имени и поставит numeric countryId сам
    # (не включаем countryId в upload JSON)
    address: Optional[str] = None
    hasDorm: Optional[bool] = False

    enrollmentDocuments: List[CreateInstitutionEnrollmentDocumentDto]
    enrollmentRequirements: List[CreateInstitutionEnrollmentRequirementDto]
    majors: Optional[List[CreateInstitutionMajorDto]] = None
