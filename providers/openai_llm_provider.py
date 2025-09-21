# providers/openai_llm_provider.py
import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional

from openai import (APIConnectionError, APITimeoutError, AsyncOpenAI,
                    BadRequestError, InternalServerError, RateLimitError)

from domain.enums import MAJOR_CATEGORY

# =========================
# Utilities
# =========================


def _to_jsonable(obj):
    import enum
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, enum.Enum):
        return obj.value
    return obj


def _schema_for_prompt(json_schema: dict) -> str:
    inner = json_schema.get("schema", json_schema)
    return json.dumps(_to_jsonable(inner), ensure_ascii=False)


def _safe_json_loads(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s or "{}")
    except Exception:
        # Last-resort: try to extract first {...} block
        try:
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(s[start:end+1])
        except Exception:
            pass
        return {}


def _norm(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


# strip parenthetical aliases like "(АГА)"
_ALIAS_PARENS = re.compile(r"\s*\([^)]*\)\s*")


def normalize_university_query(name: str) -> str:
    # Drop parenthetical aliases, trim quotes, collapse spaces
    n = _ALIAS_PARENS.sub(" ", name)
    n = n.replace("«", " ").replace("»", " ").replace('"', " ")
    return _norm(n)

# =========================
# JSON Schemas
# =========================


MAJOR_CATEGORY_VALUES = [m.value for m in MAJOR_CATEGORY]

CITIES_SCHEMA = {
    "name": "CitiesPayload",
    "schema": {
        "type": "object",
        "properties": {
            "cities": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["cities"],
        "additionalProperties": False
    },
    "strict": True
}

UNIS_SCHEMA = {
    "name": "UniversitiesPayload",
    "schema": {
        "type": "object",
        "properties": {
            "universities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "shortName": {"type": ["string", "null"]},
                        "description": {"type": ["string", "null"]},
                        "website": {"type": ["string", "null"]},
                        "foundationYear": {"type": ["string", "null"]},
                        "address": {"type": ["string", "null"]},
                        "email": {"type": ["string", "null"]},
                        "contactNumber": {"type": ["string", "null"]}
                    },
                    "required": ["name"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["universities"],
        "additionalProperties": False
    },
    "strict": True
}

MAJORS_ADMISSIONS_SCHEMA = {
    "name": "MajorsAdmissionsPayload",
    "schema": {
        "type": "object",
        "properties": {
            "majors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "durationYears": {"type": ["integer", "null"]},
                        "learningLanguage": {"type": ["string", "null"]},
                        "description": {"type": ["string", "null"]},
                        "price": {"type": ["number", "null"]},
                        "category": {"type": "string", "enum": MAJOR_CATEGORY_VALUES},
                        "requirements": {"type": ["array", "null"], "items": {"type": "string"}}
                    },
                    "required": ["name", "category"],
                    "additionalProperties": False
                }
            },
            "enrollmentDocuments": {"type": "array", "items": {"type": "string"}},
            "enrollmentRequirements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string", "enum": ["ACADEMIC", "LANGUAGE", "OTHER"]},
                        "value": {"type": ["string", "null"]}
                    },
                    "required": ["name", "type"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["majors", "enrollmentDocuments", "enrollmentRequirements"],
        "additionalProperties": False
    },
    "strict": True
}

RESOLVE_UNI_SCHEMA = {
    "name": "ResolveUniversityPayload",
    "schema": {
        "type": "object",
        "properties": {
            "name": {"type": ["string", "null"]},
            "shortName": {"type": ["string", "null"]},
            "city": {"type": ["string", "null"]},
            "website": {"type": ["string", "null"]},
            "description": {"type": ["string", "null"]},
            "email": {"type": ["string", "null"]},
            "contactNumber": {"type": ["string", "null"]},
            "altNames": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["name"],
        "additionalProperties": False
    },
    "strict": True
}

CITY_INFER_SCHEMA = {
    "name": "CityInferPayload",
    "schema": {
        "type": "object",
        "properties": {
            "city": {"type": ["string", "null"]},
            "cityCandidates": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["city", "cityCandidates"],
        "additionalProperties": False
    },
    "strict": True
}

# =========================
# Instructions
# =========================

SYS_CITIES = (
    "Ты — точный исследователь с доступом к веб-поиску. Верни только те города данной страны, "
    "в которых подтверждённо есть хотя бы одно аккредитованное высшее учебное заведение. "
    "Если не уверен — не включай. Все текстовые значения — строго на русском языке. "
    "Отвечай только JSON по заданной схеме."
)

SYS_UNIS = (
    "Ты — точный исследователь с доступом к веб-поиску. Для указанного города и страны перечисли "
    "аккредитованные высшие учебные заведения (университеты/институты/академии), расположенные в этом городе. "
    "Для каждого ВУЗа укажи официальное название, краткое описание (1–2 предложения), веб-сайт, email и телефон, если доступны. "
    "Если после проверки официальных источников данных нет — поставь null. "
    "Все текстовые значения — строго на русском языке (кроме enum). "
    "Отвечай только JSON по заданной схеме."
)

SYS_MAJORS = (
    "Ты — точный исследователь с доступом к веб-поиску. Для указанного ВУЗа извлеки РЕАЛЬНЫЕ образовательные программы/направления "
    "и РЕАЛЬНЫЕ документы/требования для поступления с официальных источников (сайт ВУЗа, министерские реестры). "
    "Верни ПОЛНЫЙ список, если возможно; допускается пустой список, если подтверждённых данных нет. "
    "Каждая программа должна быть на русском языке, а категория — одно из значений enum (на английском). "
    "Если не можешь подтвердить пункт — пропусти. "
    "Найди все актуальные специальности в данном университете. "
    "Все текстовые значения — строго на русском языке (кроме enum). "
    "Поле Price (цена) если поступление на специальность платная должна быть в KZT (Казахстанской валюте Тенге) для всех стран и университетов, только в эквиваленте Казахстанской тенге KZT"
    "Отвечай только JSON по заданной схеме."
)

SYS_RESOLVE = (
    "Ты — точный исследователь с доступом к веб-поиску. По названию ВУЗа и стране найди официальный ВУЗ: "
    "уточни актуальное официальное название, город (если можно определить), официальный сайт, контактные данные. "
    "Используй первичные источники (официальный сайт ВУЗа, государственные реестры). "
    "Если город определить нельзя надёжно — верни city = null (не придумывай). "
    "Все текстовые значения — строго на русском языке. "
    "Отвечай только JSON по заданной схеме."
)

SYS_CITY_INFER = (
    "Определи город расположения ВУЗа по официальным источникам: раздел 'Контакты', почтовый адрес, документы об аккредитации. "
    "Верни city (если однозначно определён) и cityCandidates (до 5 вариантов). "
    "Текст — строго на русском языке. Отвечай только JSON по схеме."
)

# =========================
# Provider
# =========================


class OpenAILLMProvider:
    """
    Fault-tolerant wrapper around OpenAI Responses/Chat APIs.
    - Never raises to callers; always returns safe structures.
    - Uses web_search tool when allowed.
    - Enforces timeouts and small internal retries without leaking RetryError.
    """

    def __init__(self, web_model: Optional[str] = None, fallback_model: Optional[str] = None, debug: bool = False):
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is required")
        self.client = AsyncOpenAI(api_key=key)

        self.web_model = web_model or os.getenv(
            "OPENAI_WEB_MODEL", "gpt-5-mini")
        self.fallback_model = fallback_model or os.getenv(
            "OPENAI_FALLBACK_MODEL", "gpt-5-mini")
        self.debug = debug or (os.getenv("DEBUG_LLM", "").lower() in {
                               "1", "true", "yes"})
        self.use_web = (os.getenv("USE_WEB_SEARCH", "1").lower()
                        in {"1", "true", "yes"})

        # Tunables
        self.req_timeout = float(os.getenv("LLM_TIMEOUT_SEC", "60"))
        self.layer_retries = int(
            os.getenv("LLM_LAYER_RETRIES", "2"))  # retries per layer
        # e.g., original + normalized
        self.overall_variants = int(os.getenv("LLM_OVERALL_VARIANTS", "2"))

    # -------- Internals --------

    async def _with_timeout(self, coro):
        return await asyncio.wait_for(coro, timeout=self.req_timeout)

    def _extract_text(self, resp) -> str:
        # Responses API
        txt = getattr(resp, "output_text", "") or ""
        if not txt and getattr(resp, "output", None):
            try:
                txt = resp.output[0].content[0].text
            except Exception:
                txt = ""
        # Chat Completions
        if not txt and getattr(resp, "choices", None):
            try:
                txt = resp.choices[0].message.content or ""
            except Exception:
                txt = ""
        return txt

    async def _responses_call(self, system: str, user: str, json_schema: Dict[str, Any], use_tools: bool) -> Dict[str, Any]:
        """
        One call to Responses API (optionally with tools). Safe: catches OAI exceptions.
        """
        kwargs = {
            "model": self.web_model if use_tools else self.fallback_model,
            "input": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": json_schema},
            "temperature": 0,
        }
        if use_tools and self.use_web:
            kwargs["tools"] = [{"type": "web_search"}]
            kwargs["tool_choice"] = "auto"

        try:
            resp = await self._with_timeout(self.client.responses.create(**kwargs))
            text = self._extract_text(resp)
            if self.debug:
                print("\n--- RESPONSES (tools=%s) ---\n%s\n--- END ---\n" %
                      (use_tools, (text or "")[:1500]))
            return _safe_json_loads(text)
        except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError, BadRequestError, Exception) as e:
            if self.debug:
                print(
                    f"[WARN] Responses call failed (tools={use_tools}): {type(e).__name__}: {e}")
            return {}

    async def _chat_call(self, system: str, user: str, json_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback to Chat Completions with response_format=json_object.
        """
        schema_text = _schema_for_prompt(json_schema)
        try:
            resp = await self._with_timeout(
                self.client.chat.completions.create(
                    model=self.fallback_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user +
                            "\nВерни ТОЛЬКО один JSON-объект, соответствующий этой JSON-схеме:\n" + schema_text},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                )
            )
            text = self._extract_text(resp)
            if self.debug:
                print("\n--- CHAT ---\n%s\n--- END ---\n" %
                      ((text or "")[:1500]))
            return _safe_json_loads(text)
        except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError, BadRequestError, Exception) as e:
            if self.debug:
                print(f"[WARN] Chat call failed: {type(e).__name__}: {e}")
            return {}

    async def _json_schema_safe(self, system: str, user: str, json_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Multi-layer, small retries; never raises.
        Layer 1: Responses + web_search (if enabled)
        Layer 2: Responses (no tools)
        Layer 3: Chat Completions
        """
        # tiny helper to retry a layer
        async def _try_layer(fn, attempts: int) -> Dict[str, Any]:
            last: Dict[str, Any] = {}
            for i in range(attempts):
                out = await fn()
                if out:
                    return out
                await asyncio.sleep(min(1.0 * (i + 1), 3.0))  # small backoff
                last = out
            return last

        # Layer 1
        out = await _try_layer(lambda: self._responses_call(system, user, json_schema, use_tools=True), self.layer_retries)
        if out:
            return out
        # Layer 2
        out = await _try_layer(lambda: self._responses_call(system, user, json_schema, use_tools=False), self.layer_retries)
        if out:
            return out
        # Layer 3
        out = await _try_layer(lambda: self._chat_call(system, user, json_schema), max(1, self.layer_retries - 1))
        return out or {}

    # -------- Public methods (safe) --------

    async def list_cities(self, country_name: str) -> List[str]:
        user = (
            f"Страна: {country_name}\n"
            "Верни список городов/посёлков, где есть хотя бы одно аккредитованное высшее учебное заведение. "
            "Без дубликатов и районов; если в достоверности города есть сомнения — не включай."
        )
        out = await self._json_schema_safe(SYS_CITIES, user, CITIES_SCHEMA)
        cities = out.get("cities") or []
        dedup, seen = [], set()
        for c in cities:
            cc = _norm(c)
            if cc and cc.lower() not in seen:
                seen.add(cc.lower())
                dedup.append(cc)
        return dedup

    async def list_universities(self, country_name: str, city_name: str) -> List[Dict[str, Any]]:
        user = (
            f"Страна: {country_name}\nГород: {city_name}\n"
            "Перечисли ВСЕ аккредитованные ВУЗы в этом городе (университеты/институты/академии). "
            "Поля для каждого ВУЗа: name, shortName, description, website, foundationYear (YYYY или null), address, email, contactNumber. "
            "Все текстовые значения — на русском (кроме enum)."
        )
        out = await self._json_schema_safe(SYS_UNIS, user, UNIS_SCHEMA)
        return out.get("universities") or []

    async def majors_and_admissions(self, country_name: str, city_name: Optional[str], university_name: str, website: Optional[str]) -> Dict[str, Any]:
        user = (
            f"Страна: {country_name}\nГород: {city_name or 'неизвестно'}\nВУЗ: {university_name}\n"
            f"Официальный сайт: {website or 'неизвестно'}\n"
            "Верни учебные программы (если подтверждены; допускается пустой список) и документы/требования для поступления. "
            "Текст — на русском (кроме enum)."
        )
        out = await self._json_schema_safe(SYS_MAJORS, user, MAJORS_ADMISSIONS_SCHEMA)
        out.setdefault("majors", [])
        out.setdefault("enrollmentDocuments", [])
        out.setdefault("enrollmentRequirements", [])
        return out

    async def resolve_university(self, country_name: str, query_name: str) -> Dict[str, Optional[str]]:
        """
        Resolve by name across the whole country (canonical RU name, city if known, website, contacts).
        Never raises; city may be None.
        Tries original name; if missing critical fields, tries normalized name.
        """
        variants = [query_name]
        norm_query = normalize_university_query(query_name)
        if norm_query and norm_query != query_name:
            variants.append(norm_query)

        best: Dict[str, Optional[str]] = {}
        for idx, q in enumerate(variants[: max(1, self.overall_variants)]):
            user = (
                f"Страна: {country_name}\n"
                f"Название ВУЗа (как в запросе): {q}\n"
                "Найди официальный ВУЗ в этой стране; верни актуальное официальное название (на русском), город (если известен), сайт, контактные данные. "
                "Используй только официальные источники/реестры. Если уверен в названии, но город неясен — верни city = null."
            )
            out = await self._json_schema_safe(SYS_RESOLVE, user, RESOLVE_UNI_SCHEMA)
            cand = {
                "name": (out.get("name") or None),
                "shortName": (out.get("shortName") or None),
                "city": (out.get("city") or None),
                "website": (out.get("website") or None),
                "description": (out.get("description") or None),
                "email": (out.get("email") or None),
                "contactNumber": (out.get("contactNumber") or None),
            }
            # Choose the "best" candidate by presence of name + website + city
            if (cand.get("name") and (cand.get("website") or cand.get("city"))) or idx == len(variants) - 1:
                best = cand
                if cand.get("name"):
                    break
        return best or {"name": None, "shortName": None, "city": None, "website": None, "description": None, "email": None, "contactNumber": None}

    async def find_city_candidates(self, country_name: str, university_name: str, website: Optional[str]) -> Dict[str, Any]:
        user = (
            f"Страна: {country_name}\nВУЗ: {university_name}\n"
            f"Официальный сайт: {website or 'неизвестно'}\n"
            "Определи город по официальным контактам/адресам (если невозможно — верни null и список кандидатур)."
        )
        out = await self._json_schema_safe(SYS_CITY_INFER, user, CITY_INFER_SCHEMA)
        out.setdefault("city", None)
        out.setdefault("cityCandidates", [])
        return out
