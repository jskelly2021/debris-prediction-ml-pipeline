
from enum import Enum

class TuneMode(Enum):
    """Supported hyperparameter tuning strategies."""

    NONE = 0
    RANDOM_SEARCH = 1
    GRID_SEARCH = 2


def parse_tune_mode(value) -> TuneMode:
    """Convert a config tune-mode value into a TuneMode enum."""

    if isinstance(value, TuneMode):
        return value

    normalized = str(value).strip().lower()
    mode_by_value = {
        "none": TuneMode.NONE,
        "random_search": TuneMode.RANDOM_SEARCH,
        "grid_search": TuneMode.GRID_SEARCH,
    }

    if normalized not in mode_by_value:
        valid_values = ", ".join(mode_by_value)
        raise ValueError(f"Unsupported tune mode: {value}. Expected one of: {valid_values}.")

    return mode_by_value[normalized]
