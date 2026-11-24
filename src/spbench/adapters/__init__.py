"""Adapters over upstream attention implementations."""

from spbench.adapters.base import Availability
from spbench.adapters.distflashattn import DistFlashAttnAdapter
from spbench.adapters.reference import ReferenceAdapter
from spbench.adapters.ring import RingAdapter
from spbench.adapters.ulysses import UlyssesAdapter
from spbench.adapters.usp import UspAdapter

ADAPTERS = {
    "reference": ReferenceAdapter(),
    "ring": RingAdapter(),
    "distflashattn": DistFlashAttnAdapter(),
    "ulysses": UlyssesAdapter(),
    "usp": UspAdapter(),
}

__all__ = ["ADAPTERS", "Availability"]
