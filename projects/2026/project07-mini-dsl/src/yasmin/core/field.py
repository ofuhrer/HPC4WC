from dataclasses import dataclass

from yasmin.core.dimensions import Dimension
from yasmin.core.dtype import DType


@dataclass(frozen=True, slots=True)
class Field:
    name: str
    dims: tuple[Dimension, ...]
    dtype: DType
