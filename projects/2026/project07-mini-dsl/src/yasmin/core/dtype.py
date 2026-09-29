from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DType:
    name: str

    def __str__(self) -> str:
        return self.name


float32 = DType("float32")
float64 = DType("float64")
int32 = DType("int32")
int64 = DType("int64")

_DTYPES = {
    "float32": float32,
    "float64": float64,
    "int32": int32,
    "int64": int64,
}


def dtype(value: str | DType) -> DType:
    if isinstance(value, DType):
        return value
    try:
        return _DTYPES[value]
    except KeyError as e:
        raise ValueError(f"Unsupported dtype: {value!r}") from e
