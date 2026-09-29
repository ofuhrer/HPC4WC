from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Dimension:
    name: str
