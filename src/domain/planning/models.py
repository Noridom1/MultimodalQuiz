from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class QuizPlanDomainResult:
    plans: list[Any]
    planner_name: str
