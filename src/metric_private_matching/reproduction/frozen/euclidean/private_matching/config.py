"""Validated scientific and execution configuration."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from private_matching.seeds import BIT_GENERATOR_NAME, SEED_SCHEME_VERSION

SOURCE_API_VERSION = "1"
AMBIENT_SERVER_RULE = "euclidean_report_distances"
FOLDED_SERVER_RULE = "euclidean_report_distances"
PRIMARY_MECHANISMS = ("ambient", "phi_infinity", "folded")
POINT_CONFIG_LABELS = (
    "Point configuration 1",
    "Point configuration 2",
    "Point configuration 3",
)

PRODUCTION_EXPERIMENT_NAME = "uniform_cube_three_mechanisms_configwise_v2"
PRODUCTION_N_STEP = 5
PRODUCTION_D_STEP = 5
PRODUCTION_S = 3


class ConfigError(ValueError):
    """Invalid scientific or execution configuration."""


def point_config_label(s: int) -> str:
    if 0 <= int(s) < len(POINT_CONFIG_LABELS):
        return POINT_CONFIG_LABELS[int(s)]
    return f"Point configuration {int(s) + 1}"


def point_config_dirname(s: int) -> str:
    return f"point_config_{int(s) + 1:02d}"


def parse_grid(spec: Any, *, name: str) -> list[int]:
    if isinstance(spec, Mapping):
        missing = [k for k in ("start", "stop_inclusive", "step") if k not in spec]
        if missing:
            raise ConfigError(f"{name} range is missing {missing}")
        start = int(spec["start"])
        stop = int(spec["stop_inclusive"])
        step = int(spec["step"])
        if step <= 0:
            raise ConfigError(f"{name} step must be positive")
        if stop < start:
            raise ConfigError(f"{name} stop_inclusive must be >= start")
        values = list(range(start, stop + 1, step))
        extra = {k: spec[k] for k in spec if k not in {"start", "stop_inclusive", "step"}}
        if extra:
            raise ConfigError(f"{name} has unexpected keys {sorted(extra)}")
        return values
    if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
        values = [int(v) for v in spec]
        if not values:
            raise ConfigError(f"{name} must be non-empty")
        return values
    raise ConfigError(f"{name} must be a range mapping or a list of integers")


def unique_settings(
    n_values: Sequence[int],
    d_values: Sequence[int],
    varying_n_fixed_d: int,
    varying_d_fixed_n: int,
) -> list[tuple[int, int]]:
    settings = {(int(n), int(varying_n_fixed_d)) for n in n_values}
    settings.update((int(varying_d_fixed_n), int(d)) for d in d_values)
    return sorted(settings)


@dataclass(frozen=True)
class ScientificConfig:
    name: str
    root_seed: int
    cube_side: float
    dtype: str
    mechanisms: tuple[str, ...]
    ambient_server_rule: str
    folded_server_rule: str
    epsilons: tuple[float, ...]
    n_values: tuple[int, ...]
    d_values: tuple[int, ...]
    n_grid_step: int | None
    d_grid_step: int | None
    varying_n_fixed_d: int
    varying_d_fixed_n: int
    master_n: int
    master_d: int
    configurations_S: int
    noise_repetitions_K: int
    seed_scheme_version: str = SEED_SCHEME_VERSION
    bit_generator: str = BIT_GENERATOR_NAME
    source_api_version: str = SOURCE_API_VERSION
    plot_epsilons: tuple[float, ...] = field(default_factory=tuple)
    plot_separate_master_configurations: bool = True
    plot_x_scale: str = "linear"
    plot_y_scale: str = "linear"
    plot_confidence_level: float = 0.95
    plot_interval_method: str = "conditional_noise_mean_t"
    plot_layout: str = "sweep_columns_full_bounds_and_empirical_detail"
    rank_svd_eps_multiplier: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "plot_epsilons", tuple(self.plot_epsilons or self.epsilons))

    @property
    def dtype_name(self) -> str:
        return str(self.dtype)

    def settings(self) -> list[tuple[int, int]]:
        return unique_settings(
            self.n_values,
            self.d_values,
            self.varying_n_fixed_d,
            self.varying_d_fixed_n,
        )

    def n_sweep_settings(self) -> list[tuple[int, int]]:
        return [(int(n), int(self.varying_n_fixed_d)) for n in self.n_values]

    def d_sweep_settings(self) -> list[tuple[int, int]]:
        return [(int(self.varying_d_fixed_n), int(d)) for d in self.d_values]

    def intersection_setting(self) -> tuple[int, int]:
        return (int(self.varying_d_fixed_n), int(self.varying_n_fixed_d))

    def workload_counts(self) -> dict[str, int]:
        n_settings = len(self.settings())
        n_units = n_settings * int(self.configurations_S)
        n_mech = len(self.mechanisms)
        n_eps = len(self.epsilons)
        k = int(self.noise_repetitions_K)
        return {
            "unique_settings": n_settings,
            "configurations_S": int(self.configurations_S),
            "work_units": n_units,
            "mechanisms": n_mech,
            "epsilons": n_eps,
            "noise_repetitions_K": k,
            "mechanism_repetitions": n_units * n_mech * n_eps * k,
            "opt_solves": n_units,
            "main_figures": int(self.configurations_S) * len(self.plot_epsilons),
        }

    def scientific_identity(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_api_version": self.source_api_version,
            "seed_scheme_version": self.seed_scheme_version,
            "bit_generator": self.bit_generator,
            "root_seed": int(self.root_seed),
            "cube_side": float(self.cube_side),
            "dtype": self.dtype_name,
            "mechanisms": list(self.mechanisms),
            "ambient_server_rule": self.ambient_server_rule,
            "folded_server_rule": self.folded_server_rule,
            "master_n": int(self.master_n),
            "master_d": int(self.master_d),
            "configurations_S": int(self.configurations_S),
            "L": 1.0,
        }


@dataclass(frozen=True)
class ExecutionConfig:
    local_workers: int = 2
    numerical_threads_per_worker: int = 1
    resume: bool = True
    serial: bool = False
    results_root: str | None = None
    logs_root: str | None = None
    array_task_id: int | None = None
    array_task_count: int | None = None
    repetition_block_size: int | None = None
    write_masters: bool = True
    record_diagnostics: bool = False
    slurm: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentBundle:
    scientific: ScientificConfig
    execution: ExecutionConfig
    source_paths: tuple[str, ...] = ()

    def results_dir(self) -> Path:
        root = self.execution.results_root
        if not root:
            raise ConfigError(
                "execution.results_root is unset. Set it in the execution YAML "
                "(cluster storage path) rather than relying on a hardcoded default."
            )
        return Path(root) / self.scientific.name


def _require_positive_number(name: str, value: float) -> None:
    if not (float(value) > 0.0):
        raise ConfigError(f"{name} must be > 0, got {value!r}")


def _as_float_tuple(values: Sequence[Any], *, name: str) -> tuple[float, ...]:
    out = tuple(float(v) for v in values)
    if not out:
        raise ConfigError(f"{name} must be non-empty")
    for v in out:
        _require_positive_number(name, v)
    if len(set(out)) != len(out):
        raise ConfigError(f"{name} must not contain duplicates")
    return out


def validate_scientific(cfg: ScientificConfig, *, profile: str | None = None) -> None:
    if not cfg.name:
        raise ConfigError("experiment.name is required")
    if cfg.dtype_name != "float64":
        raise ConfigError("dtype must be float64 for the initial implementation")
    if cfg.seed_scheme_version != SEED_SCHEME_VERSION:
        raise ConfigError(
            f"unsupported seed_scheme_version {cfg.seed_scheme_version!r}; "
            f"this code implements {SEED_SCHEME_VERSION}"
        )
    if cfg.bit_generator != BIT_GENERATOR_NAME:
        raise ConfigError(
            f"unsupported bit_generator {cfg.bit_generator!r}; "
            f"this code implements {BIT_GENERATOR_NAME}"
        )
    if tuple(cfg.mechanisms) != PRIMARY_MECHANISMS:
        raise ConfigError(
            f"mechanisms must be {PRIMARY_MECHANISMS} in that order, got {cfg.mechanisms}"
        )
    if cfg.ambient_server_rule != AMBIENT_SERVER_RULE:
        raise ConfigError(
            f"ambient_server_rule must be {AMBIENT_SERVER_RULE!r} "
            f"(a different rule is a new scientific identity)"
        )
    if cfg.folded_server_rule != FOLDED_SERVER_RULE:
        raise ConfigError(
            f"folded_server_rule must be {FOLDED_SERVER_RULE!r}; "
            "do not silently substitute folded_debiased"
        )
    _require_positive_number("cube_side", cfg.cube_side)
    if abs(float(cfg.cube_side) - 1.0) > 0.0:
        raise ConfigError("cube_side must be 1.0 in the main experiment (L=1)")
    if int(cfg.configurations_S) < 1:
        raise ConfigError("configurations_S must be >= 1")
    if int(cfg.configurations_S) > PRODUCTION_S:
        raise ConfigError(
            f"configurations_S={cfg.configurations_S} exceeds {PRODUCTION_S}; "
            "do not increase S without explicit user instruction"
        )
    if int(cfg.noise_repetitions_K) < 1:
        raise ConfigError("noise_repetitions_K must be >= 1")
    for n in cfg.n_values:
        if int(n) < 2:
            raise ConfigError(f"n values must be >= 2, got {n}")
    for d in cfg.d_values:
        if int(d) < 1:
            raise ConfigError(f"d values must be >= 1, got {d}")
    if int(cfg.varying_n_fixed_d) < 1 or int(cfg.varying_d_fixed_n) < 2:
        raise ConfigError("fixed slice values are invalid")
    if int(cfg.varying_d_fixed_n) not in set(cfg.n_values):
        raise ConfigError(
            f"varying_d_fixed_n={cfg.varying_d_fixed_n} must appear in n_values "
            "so the (n,d) intersection is on the n-sweep grid"
        )
    if int(cfg.varying_n_fixed_d) not in set(cfg.d_values):
        raise ConfigError(
            f"varying_n_fixed_d={cfg.varying_n_fixed_d} must appear in d_values "
            "so the (n,d) intersection is on the d-sweep grid"
        )
    max_n = max(max(cfg.n_values), cfg.varying_d_fixed_n)
    max_d = max(max(cfg.d_values), cfg.varying_n_fixed_d)
    if int(cfg.master_n) < max_n:
        raise ConfigError(f"master_n={cfg.master_n} is smaller than max n={max_n}")
    if int(cfg.master_d) < max_d:
        raise ConfigError(f"master_d={cfg.master_d} is smaller than max evaluated d={max_d}")
    if max_d > 500 and profile == "production":
        raise ConfigError(
            "evaluated d grid exceeds 500; extending toward master_d=1000 requires "
            "a reviewed, approved workload"
        )
    plot_eps = set(cfg.plot_epsilons)
    sim_eps = set(cfg.epsilons)
    missing = plot_eps - sim_eps
    if missing:
        raise ConfigError(f"plot epsilons {sorted(missing)} were not simulated")
    if cfg.plot_x_scale != "linear" or cfg.plot_y_scale != "linear":
        raise ConfigError("plot axes must be linear")
    if cfg.plot_interval_method != "conditional_noise_mean_t":
        raise ConfigError("plot interval_method must be conditional_noise_mean_t")
    if not cfg.plot_separate_master_configurations:
        raise ConfigError("master configurations must be plotted separately")
    if profile in {"production", "pilot"}:
        if cfg.name != PRODUCTION_EXPERIMENT_NAME:
            raise ConfigError(
                f"{profile} experiment.name must remain {PRODUCTION_EXPERIMENT_NAME!r} "
                "so compatible work can be reused"
            )
        if int(cfg.configurations_S) != PRODUCTION_S:
            raise ConfigError(f"{profile} must use configurations_S={PRODUCTION_S}")
        if int(cfg.master_n) != 250 or int(cfg.master_d) != 1000:
            raise ConfigError(f"{profile} must freeze master_n=250 and master_d=1000")
        if int(cfg.root_seed) != 20260914:
            raise ConfigError(f"{profile} must use root_seed=20260914")
    if profile == "production":
        if cfg.n_grid_step != PRODUCTION_N_STEP or cfg.d_grid_step != PRODUCTION_D_STEP:
            raise ConfigError("production grids must use step 5; do not replace with log spacing")
        if max(cfg.d_values) > 500:
            raise ConfigError("do not simulate d>500 automatically in production")
        if tuple(cfg.epsilons) != (1.0, 8.0, 16.0):
            raise ConfigError("production epsilons must be [1, 8, 16]")
        if tuple(cfg.plot_epsilons) != (1.0, 8.0, 16.0):
            raise ConfigError("production plot epsilons must be [1, 8, 16]")
        if int(cfg.noise_repetitions_K) != 100:
            raise ConfigError("production K must be 100 unless the user revises it")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping")
    return data


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def scientific_from_mapping(data: Mapping[str, Any]) -> ScientificConfig:
    exp = dict(data.get("experiment") or {})
    plot = dict(data.get("plot") or {})
    if "n_values" not in exp or "d_values" not in exp:
        raise ConfigError("experiment.n_values and experiment.d_values are required")
    n_spec = exp["n_values"]
    d_spec = exp["d_values"]
    n_values = parse_grid(n_spec, name="n_values")
    d_values = parse_grid(d_spec, name="d_values")
    n_step = int(n_spec["step"]) if isinstance(n_spec, Mapping) else None
    d_step = int(d_spec["step"]) if isinstance(d_spec, Mapping) else None
    mechanisms = tuple(exp.get("mechanisms") or PRIMARY_MECHANISMS)
    cfg = ScientificConfig(
        name=str(exp["name"]),
        root_seed=int(exp["root_seed"]),
        cube_side=float(exp.get("cube_side", 1.0)),
        dtype=str(exp.get("dtype", "float64")),
        mechanisms=mechanisms,
        ambient_server_rule=str(exp.get("ambient_server_rule", AMBIENT_SERVER_RULE)),
        folded_server_rule=str(exp.get("folded_server_rule", FOLDED_SERVER_RULE)),
        epsilons=_as_float_tuple(exp.get("epsilons") or (1, 8, 16), name="epsilons"),
        n_values=tuple(n_values),
        d_values=tuple(d_values),
        n_grid_step=n_step,
        d_grid_step=d_step,
        varying_n_fixed_d=int(exp["varying_n_fixed_d"]),
        varying_d_fixed_n=int(exp["varying_d_fixed_n"]),
        master_n=int(exp["master_n"]),
        master_d=int(exp["master_d"]),
        configurations_S=int(exp["configurations_S"]),
        noise_repetitions_K=int(exp["noise_repetitions_K"]),
        seed_scheme_version=str(exp.get("seed_scheme_version", SEED_SCHEME_VERSION)),
        bit_generator=str(exp.get("bit_generator", BIT_GENERATOR_NAME)),
        source_api_version=str(exp.get("source_api_version", SOURCE_API_VERSION)),
        plot_epsilons=_as_float_tuple(plot.get("epsilons") or exp.get("epsilons") or (1, 8, 16), name="plot.epsilons"),
        plot_separate_master_configurations=bool(plot.get("separate_master_configurations", True)),
        plot_x_scale=str(plot.get("x_scale", "linear")),
        plot_y_scale=str(plot.get("y_scale", "linear")),
        plot_confidence_level=float(plot.get("confidence_level", 0.95)),
        plot_interval_method=str(plot.get("interval_method", "conditional_noise_mean_t")),
        plot_layout=str(plot.get("layout", "sweep_columns_full_bounds_and_empirical_detail")),
        rank_svd_eps_multiplier=float(exp.get("rank_svd_eps_multiplier", 1.0)),
    )
    return cfg


def execution_from_mapping(data: Mapping[str, Any]) -> ExecutionConfig:
    exe = dict(data.get("execution") or {})
    allowed = {f.name for f in fields(ExecutionConfig)}
    unknown = set(exe) - allowed
    if unknown:
        raise ConfigError(f"execution has unexpected keys {sorted(unknown)}")
    slurm = dict(exe.get("slurm") or {})
    return ExecutionConfig(
        local_workers=int(exe.get("local_workers", 2)),
        numerical_threads_per_worker=int(exe.get("numerical_threads_per_worker", 1)),
        resume=bool(exe.get("resume", True)),
        serial=bool(exe.get("serial", False)),
        results_root=exe.get("results_root"),
        logs_root=exe.get("logs_root"),
        array_task_id=exe.get("array_task_id"),
        array_task_count=exe.get("array_task_count"),
        repetition_block_size=exe.get("repetition_block_size"),
        write_masters=bool(exe.get("write_masters", True)),
        record_diagnostics=bool(exe.get("record_diagnostics", False)),
        slurm=slurm,
    )


def load_bundle(
    paths: Sequence[str | Path],
    *,
    profile: str | None = None,
    execution_overrides: Mapping[str, Any] | None = None,
) -> ExperimentBundle:
    merged: dict[str, Any] = {}
    resolved_paths: list[str] = []
    for raw in paths:
        path = Path(raw).resolve()
        merged = _deep_merge(merged, _load_yaml(path))
        resolved_paths.append(str(path))
    if execution_overrides:
        merged = _deep_merge(merged, {"execution": dict(execution_overrides)})
    scientific = scientific_from_mapping(merged)
    validate_scientific(scientific, profile=profile)
    execution = execution_from_mapping(merged)
    if int(execution.local_workers) < 1:
        raise ConfigError("local_workers must be >= 1")
    if int(execution.numerical_threads_per_worker) < 1:
        raise ConfigError("numerical_threads_per_worker must be >= 1")
    if execution.repetition_block_size is not None and int(execution.repetition_block_size) < 1:
        raise ConfigError("repetition_block_size must be >= 1 when set")
    return ExperimentBundle(
        scientific=scientific,
        execution=execution,
        source_paths=tuple(resolved_paths),
    )


def resolved_table(bundle: ExperimentBundle) -> list[tuple[str, str]]:
    sci = bundle.scientific
    exe = bundle.execution
    counts = sci.workload_counts()
    rows = [
        ("experiment.name", sci.name),
        ("root_seed", str(sci.root_seed)),
        ("seed_scheme_version", sci.seed_scheme_version),
        ("bit_generator", sci.bit_generator),
        ("cube_side / L", str(sci.cube_side)),
        ("dtype", sci.dtype_name),
        ("mechanisms", ", ".join(sci.mechanisms)),
        ("ambient_server_rule", sci.ambient_server_rule),
        ("folded_server_rule", sci.folded_server_rule),
        ("epsilons", ", ".join(epsilon_key(e) for e in sci.epsilons)),
        ("plot_epsilons", ", ".join(epsilon_key(e) for e in sci.plot_epsilons)),
        ("n_values", _grid_summary(sci.n_values, sci.n_grid_step)),
        ("d_values", _grid_summary(sci.d_values, sci.d_grid_step)),
        ("varying_n_fixed_d", str(sci.varying_n_fixed_d)),
        ("varying_d_fixed_n", str(sci.varying_d_fixed_n)),
        ("master_n", str(sci.master_n)),
        ("master_d", str(sci.master_d)),
        ("evaluated_d_max", str(max(sci.d_values))),
        ("configurations_S", str(sci.configurations_S)),
        ("noise_repetitions_K", str(sci.noise_repetitions_K)),
        ("unique_(n,d)_settings", str(counts["unique_settings"])),
        ("work_units", str(counts["work_units"])),
        ("mechanism_repetitions", str(counts["mechanism_repetitions"])),
        ("opt_solves", str(counts["opt_solves"])),
        ("main_figures", str(counts["main_figures"])),
        ("local_workers", str(exe.local_workers)),
        ("numerical_threads_per_worker", str(exe.numerical_threads_per_worker)),
        ("resume", str(exe.resume)),
        ("serial", str(exe.serial)),
        ("results_root", str(exe.results_root)),
        ("logs_root", str(exe.logs_root)),
    ]
    return rows


def _grid_summary(values: Sequence[int], step: int | None) -> str:
    vals = list(values)
    if step is not None and vals:
        return f"{vals[0]}..{vals[-1]} step {step} ({len(vals)} values)"
    return f"{vals} ({len(vals)} values)"


def epsilon_key(epsilon: float) -> str:
    """Stable map from a numeric budget to a dict/JSON key."""
    value = float(epsilon)
    if not (value > 0.0):
        raise ConfigError(f"epsilon must be > 0, got {epsilon!r}")
    if value == int(value) and abs(value) < 1e16:
        return str(int(value))
    return format(value, ".16g")
