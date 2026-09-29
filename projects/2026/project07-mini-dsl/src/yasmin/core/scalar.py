from dataclasses import dataclass

from yasmin.core.dtype import DType


@dataclass(frozen=True, slots=True)
class Scalar:
    name: str
    dtype: DType
