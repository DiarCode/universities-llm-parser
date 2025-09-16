# main.py
import argparse
import asyncio
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from tqdm.asyncio import tqdm

from app_io.writers import (
    write_country_city_json,
    write_preflight_report,
    write_universities_json,
)
from domain.dto import (
    CityRow,
    CountryRow,
    CreateInstitutionBulkDto,
    CreateInstitutionEnrollmentDocumentDto,
    CreateInstitutionEnrollmentRequirementDto,
    CreateInstitutionMajorDto,
)
from providers.openai_llm_provider import OpenAILLMProvider
from services.normalize import (
    country_meta_from_code,
    infer_financing,
    infer_type,
    resolve_country_code,
)

load_dotenv()

OUT_DIR = Path(os.getenv("OUT_DIR", "out"))
CONCURRENCY = int(os.getenv("CONCURRENCY", "8"))
ALLOW_UNKNOWN_CITY = os.getenv("ALLOW_UNKNOWN_CITY", "1").lower() in {
    "1", "true", "yes"}

_norm_ws = re.compile(r"\s+")


def norm(s: str) -> str:
    return _norm_ws.sub(" ", (s or "").strip().lower())


def load_names_file(path: Path) -> List[str]:
    names: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            names.append(line)
    # unique, stable
    seen: Set[str] = set()
    uniq: List[str] = []
    for n in names:
        k = norm(n)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(n)
    return uniq


async def _build_one_university(
    provider: OpenAILLMProvider,
    country_name: str,
    query_name: str,
) -> Tuple[CreateInstitutionBulkDto, Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Always returns a DTO (best-effort) + resolver payload (if any) + preflight notes.
    Nothing raises; majors/docs/requirements may be empty arrays.
    """
    notes: List[Dict[str, Any]] = []

    # 1) Resolve (robust, never raises)
    resolved = await provider.resolve_university(country_name, query_name)
    canon_name = (resolved.get("name") or "").strip() or query_name.strip()
    city_name = (resolved.get("city") or "").strip()
    website = (resolved.get("website") or None)

    if not resolved.get("name"):
        notes.append({"university": query_name,
                     "reason": "resolver_missing_name_used_query"})

    # 2) City inference fallback
    if not city_name:
        guess = await provider.find_city_candidates(country_name, canon_name, website)
        if guess.get("city"):
            city_name = (guess["city"] or "").strip()
            notes.append({"university": canon_name,
                         "reason": "city_inferred_from_contacts"})
        elif guess.get("cityCandidates"):
            city_name = (guess["cityCandidates"][0] or "").strip()
            notes.append({"university": canon_name, "reason": "city_candidate_selected",
                         "candidates": guess["cityCandidates"][:5]})
        elif ALLOW_UNKNOWN_CITY:
            city_name = "Город не указан"
            notes.append({"university": canon_name,
                         "reason": "city_unknown_placeholder"})
        else:
            notes.append({"university": canon_name,
                         "reason": "city_unresolved"})

    # 3) Majors & admissions (safe defaults)
    details = await provider.majors_and_admissions(country_name, city_name or None, canon_name, website)
    majors_raw = details.get("majors") or []
    docs_raw = details.get("enrollmentDocuments") or []
    reqs_raw = details.get("enrollmentRequirements") or []

    majors: List[Dict[str, Any]] = []
    for m in majors_raw:
        try:
            majors.append(CreateInstitutionMajorDto(**m).model_dump())
        except Exception:
            continue

    docs: List[Dict[str, Any]] = []
    for d in docs_raw:
        if isinstance(d, str) and d.strip():
            docs.append(CreateInstitutionEnrollmentDocumentDto(
                name=d.strip()).model_dump())

    reqs: List[Dict[str, Any]] = []
    for r in reqs_raw:
        try:
            reqs.append(
                CreateInstitutionEnrollmentRequirementDto(
                    **{"name": r.get("name"), "type": r.get("type"), "value": r.get("value")}
                ).model_dump()
            )
        except Exception:
            continue

    # 4) DTO (never fail)
    dto = CreateInstitutionBulkDto(
        name=canon_name,
        shortName=resolved.get("shortName"),
        description=resolved.get("description"),
        foundationYear=None,
        financingType=infer_financing(canon_name).value,
        type=infer_type(canon_name).value,
        website=website,
        email=resolved.get("email"),
        contactNumber=resolved.get("contactNumber"),
        cityId=city_name or "Город не указан",           # string, may be placeholder
        address=resolved.get("city") or city_name or "Город не указан",
        hasDorm=False,
        enrollmentDocuments=[
            CreateInstitutionEnrollmentDocumentDto(**d)
            if isinstance(d, dict)
            else CreateInstitutionEnrollmentDocumentDto(name=d)
            for d in (docs or [])
        ],
        enrollmentRequirements=[
            CreateInstitutionEnrollmentRequirementDto(**r) for r in (reqs or [])
        ],
        majors=[CreateInstitutionMajorDto(**m) for m in (majors or [])] or [],
    )
    return dto, resolved, notes


async def main():
    parser = argparse.ArgumentParser(
        description="Universities pipeline (names-only): resolve → infer city (if needed) → majors (optional) → JSON."
    )
    parser.add_argument("--country", required=True,
                        help="Country name (e.g., Kazakhstan) or ISO2 (e.g., KZ)")
    parser.add_argument("--universities-file", type=Path, required=True,
                        help="Path to .txt with one university per line.")
    args = parser.parse_args()

    code, country_name_resolved = resolve_country_code(args.country)
    country_name_resolved, emoji = country_meta_from_code(code)
    country = CountryRow(id=1, name=country_name_resolved, emoji=emoji)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    provider = OpenAILLMProvider()

    targets = load_names_file(args.universities_file)
    if not targets:
        raise SystemExit("universities-file has no valid names.")

    sem = asyncio.Semaphore(CONCURRENCY)
    dtos: List[CreateInstitutionBulkDto] = []
    cities_set: Set[str] = set()
    preflight: List[Dict[str, Any]] = []

    async def _one(name: str):
        nonlocal dtos, cities_set, preflight
        async with sem:
            try:
                dto, resolved, notes = await _build_one_university(provider, country_name_resolved, name)
                dtos.append(dto)
                if dto.cityId:
                    cities_set.add(dto.cityId)
                preflight.extend(notes)
            except Exception as e:
                # Absolute last-resort guard: still emit minimal DTO
                dto = CreateInstitutionBulkDto(
                    name=name,
                    shortName=None,
                    description=None,
                    foundationYear=None,
                    financingType=infer_financing(name).value,
                    type=infer_type(name).value,
                    website=None,
                    email=None,
                    contactNumber=None,
                    cityId="Город не указан",
                    address="Город не указан",
                    hasDorm=False,
                    enrollmentDocuments=[],
                    enrollmentRequirements=[],
                    majors=[],
                )
                dtos.append(dto)
                cities_set.add(dto.cityId)
                preflight.append({"university": name, "city": None,
                                 "reason": f"guard_minimal_dto:{type(e).__name__}"})

    tasks = [asyncio.create_task(_one(n)) for n in targets]
    pbar = tqdm(total=len(tasks), desc=f"Resolve & parse {country.name}")
    for f in asyncio.as_completed(tasks):
        await f
        pbar.update(1)
    pbar.close()

    # Cities JSON from emitted records
    cities_rows: List[CityRow] = []
    for i, cname in enumerate(sorted(cities_set), start=1):
        cities_rows.append(CityRow(id=i, countryId=country.id, name=cname))

    # Write outputs
    write_country_city_json(OUT_DIR, country, cities_rows)
    slug = country_name_resolved.lower().replace(" ", "-")
    unis_path = write_universities_json(OUT_DIR, slug, dtos)
    report_path = write_preflight_report(OUT_DIR, preflight)

    print(
        "OK\n"
        f" Country: {country_name_resolved} ({code})\n"
        f" Countries JSON: {OUT_DIR / 'countries.json'}\n"
        f" Cities JSON: {OUT_DIR / 'cities.json'}\n"
        f" Universities JSON: {unis_path}\n"
        f" Preflight report: {report_path}\n"
        f" Institutions: {len(dtos)}  Cities: {len(cities_rows)}"
    )

if __name__ == "__main__":
    asyncio.run(main())
