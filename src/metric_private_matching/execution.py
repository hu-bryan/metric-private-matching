"""Conservative process counts for laptops, containers, and scheduled jobs."""
from __future__ import annotations

import os
from pathlib import Path


def available_cpus():
    count = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        count = min(count, len(os.sched_getaffinity(0)))
    if os.environ.get("SLURM_CPUS_PER_TASK", "").isdigit():
        count = min(count, int(os.environ["SLURM_CPUS_PER_TASK"]))
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            count = min(count, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    return max(1, count)


def available_memory():
    values = []
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                values.append(int(line.split()[1]) * 1024)
        limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if limit != "max":
            used = int(Path("/sys/fs/cgroup/memory.current").read_text())
            values.append(max(0, int(limit) - used))
    except (OSError, ValueError):
        pass
    if not values:
        try:
            values.append(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except (AttributeError, OSError, ValueError):
            return None
    return min(values)


def workers(value="auto", *, memory_per_worker=768 * 1024**2):
    if str(value) != "auto":
        result = int(value)
        if result < 1:
            raise ValueError("workers must be 'auto' or a positive integer")
        return result
    memory = available_memory()
    # Unknown memory => sequential; never assume a large machine from CPU count.
    memory_cap = max(1, int(memory * 0.5 / memory_per_worker)) if memory else 1
    return min(4, available_cpus(), memory_cap)
