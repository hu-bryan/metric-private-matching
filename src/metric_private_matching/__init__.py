"""Metric-private matching on user-provided NumPy arrays.

Rows of ``private`` and ``public`` are points; ``permutation[i]`` is the
public partner assigned to private row i. See ``match`` for the four mechanisms.
"""
__all__ = [
    "MECHANISMS", "ProjectedReport", "match", "matching_from_costs",
    "matching_from_points", "privatize_points", "privatize_projected",
    "privatize_distances", "original_points", "projected_representation",
    "distance_vectors", "gumbel_matching",
]
__version__ = "0.1.0"


def __getattr__(name):
    # CLI help and workload plans require no numerical imports.
    if name in __all__:
        from importlib import import_module
        value = getattr(import_module(".mechanisms", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(name)
