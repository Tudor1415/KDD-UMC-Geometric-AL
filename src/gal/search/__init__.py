"""GAL search package."""

from .bounds import BallTreeBounds, BoundContext, BoundsResult, BoundsStrategy
from .kd_bounds import KdTreeBounds
from .engine import Search, search_pair
from .strategies import DiversityVisitStrategy, LowerBoundVisitStrategy, VisitStrategy

# Backwards compatibility aliases (deprecated)
AngularBounds = BallTreeBounds
Bounds = BoundsStrategy
LowerBoundObjective = LowerBoundVisitStrategy
VisitingObjective = VisitStrategy

__all__ = [
    "BallTreeBounds",
    "BoundContext",
    "BoundsResult",
    "BoundsStrategy",
    "KdTreeBounds",
    "DiversityVisitStrategy",
    "LowerBoundVisitStrategy",
    "Search",
    "VisitStrategy",
    "search_pair",
    # Deprecated aliases
    "AngularBounds",
    "Bounds",
    "LowerBoundObjective",
    "VisitingObjective",
]
