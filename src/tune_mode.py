
from enum import Enum

class TuneMode(Enum):
    """Supported hyperparameter tuning strategies."""

    NONE = 0
    RANDOM_SEARCH = 1
    GRID_SEARCH = 2
