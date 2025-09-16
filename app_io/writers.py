# app_io/writers.py
import json
from pathlib import Path
from typing import List, Dict, Any
from domain.dto import CountryRow, CityRow, CreateInstitutionBulkDto

def write_country_city_json(out_dir: Path,
                            country: CountryRow,
                            cities: List[CityRow]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "countries.json").write_text(
        json.dumps([country.model_dump()], ensure_ascii=False, indent=2), "utf-8"
    )
    (out_dir / "cities.json").write_text(
        json.dumps([c.model_dump() for c in cities], ensure_ascii=False, indent=2), "utf-8"
    )

def write_universities_json(out_dir: Path,
                            country_slug: str,
                            institutions: List[CreateInstitutionBulkDto]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{country_slug}-universities.json"
    path.write_text(
        json.dumps([i.model_dump(by_alias=True) for i in institutions], ensure_ascii=False, indent=2),
        "utf-8",
    )
    return path

def write_preflight_report(out_dir: Path, report: List[Dict[str, Any]]) -> Path:
    """
    Preflight report for potential skips (e.g., city not resolvable).
    Script can only guess — backend will do the authoritative skip.
    """
    path = out_dir / "upload_preflight_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    return path
