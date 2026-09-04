from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, TypeAlias

VarIndex: TypeAlias = int
QuboKey: TypeAlias = Tuple[VarIndex, VarIndex]


@dataclass(slots=True)
class QUBOData:
    """Container for a quadratic unconstrained binary optimization instance."""

    num_variables: int
    linear: Dict[VarIndex, float]
    quadratic: Dict[QuboKey, float]
    constant: float = 0.0
    var_metadata: Dict[VarIndex, Dict[str, Any]] = field(default_factory=dict)
    penalty_weights: Dict[str, float] = field(default_factory=dict)
