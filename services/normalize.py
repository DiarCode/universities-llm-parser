from typing import Tuple
from domain.country_map import ISO2, NAME_TO_ISO
from domain.enums import INSTITUTION_FINANCING_TYPE, INSTITUTION_TYPE

def country_meta_from_code(code: str) -> Tuple[str, str | None]:
    code = code.upper()
    if code in ISO2:
        name, emoji = ISO2[code]
        return name, emoji
    return code, None

def resolve_country_code(user_input: str) -> Tuple[str, str]:
    ui = user_input.strip()
    if len(ui) == 2 and ui.isalpha():
        code = ui.upper()
        name, _ = country_meta_from_code(code)
        return code, name
    code = NAME_TO_ISO.get(ui.lower())
    if code:
        name, _ = country_meta_from_code(code)
        return code, name
    return ui.upper(), ui

def infer_type(name: str) -> INSTITUTION_TYPE:
    n = name.lower()
    if "university" in n: return INSTITUTION_TYPE.UNIVERSITY
    if any(k in n for k in ["college","institute","academy"]): return INSTITUTION_TYPE.COLLEGE
    return INSTITUTION_TYPE.UNIVERSITY

def infer_financing(name: str) -> INSTITUTION_FINANCING_TYPE:
    n = name.lower()
    if "private" in n: return INSTITUTION_FINANCING_TYPE.PRIVATE
    if any(k in n for k in ["public","state","national","federal","gov"]): return INSTITUTION_FINANCING_TYPE.GOV
    return INSTITUTION_FINANCING_TYPE.AUTONOMOUS
