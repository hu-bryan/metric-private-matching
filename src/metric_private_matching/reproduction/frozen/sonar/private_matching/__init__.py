"""Metric-private matching simulation package.

The frozen baselines retain ``euclidean_report_distances``. The ``followup``
command adds anchor-distance, public approximations, and comparison baselines.
"""

from private_matching.thread_limits import apply_thread_limits

apply_thread_limits()

__all__ = ["apply_thread_limits"]
