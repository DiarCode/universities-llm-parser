import json
from pathlib import Path
from typing import List

class Checkpoint:
    def __init__(self, path: Path):
        self.path = path
        self.cities: List[str] = []
        self.city_ptr: int = 0
        self.done: bool = False

    def load(self):
        if self.path.exists():
            data = json.loads(self.path.read_text("utf-8"))
            self.cities = data.get("cities", [])
            self.city_ptr = data.get("city_ptr", 0)
            self.done = data.get("done", False)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "cities": self.cities,
            "city_ptr": self.city_ptr,
            "done": self.done,
        }, ensure_ascii=False, indent=2), "utf-8")
