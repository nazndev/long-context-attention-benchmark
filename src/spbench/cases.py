"""Experiment-matrix expansion and rejection rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from spbench.errors import InvalidConfiguration
from spbench.layouts import basic_usp_preserves_causal_ring_order, build_group_maps

DISTFLASH_HEAD_DIMS = frozenset({16, 32, 64, 128})
DISTFLASH_BLOCK_M = 128  # lightseq_async_attn.py at commit 8f486dad


@dataclass
class Case:
    method: str
    variant: str
    phase: str
    topology: str
    nodes: int
    world_size: int
    global_sequence_length: int
    ulysses_degree: int | None
    ring_degree: int | None
    placement: str | None
    genuine_hybrid: bool
    dtype: str
    causal: bool
    batch_size: int
    heads: int
    kv_heads: int
    head_dim: int
    hidden_size: int
    dropout: float
    status: str = "planned"
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    hypothetical: bool = False
    placement_distinct: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _reject(reasons: list[str], text: str) -> None:
    reasons.append(text)


def validate_case(case: Case) -> list[str]:
    """Return rejection reasons. An empty list means the case is legal."""

    reasons: list[str] = []
    if case.batch_size != 1:
        _reject(reasons, "primary workload batch size is 1")
    if case.dropout != 0:
        _reject(reasons, "dropout must be 0; ring-flash-attention does not support dropout")
    if case.causal is not True:
        _reject(reasons, "primary matrix is causal; noncausal controls are not launched from this config")
    if case.heads < 1 or case.head_dim < 1 or case.global_sequence_length < 1 or case.world_size < 1:
        _reject(reasons, "heads, head_dim, sequence, and world size must be positive")
    if case.kv_heads != case.heads:
        _reject(reasons, "this benchmark is MHA: kv_heads must equal heads; GQA is not implemented")
    if case.hidden_size != case.heads * case.head_dim:
        _reject(reasons, "hidden_size must equal heads * head_dim")
    if case.dtype not in {"bf16", "fp16"}:
        _reject(reasons, "dtype must be bf16, or fp16 only as a separately labeled fallback")
    if case.phase not in {"forward", "forward_backward"}:
        _reject(reasons, f"unknown phase {case.phase}")
    if case.nodes not in {1, 2}:
        _reject(reasons, "topologies are one node or two nodes")
    if case.nodes == 2 and case.world_size % 2 != 0:
        _reject(reasons, "two-node runs need an equal number of ranks per node")
    if case.method in {"single_gpu_sdpa", "single_gpu_flash"}:
        if case.world_size != 1 or case.nodes != 1:
            _reject(reasons, "the single-GPU baseline is one process on one node and is not a distributed method")
        return reasons
    if case.world_size < 2:
        _reject(reasons, f"{case.method} is distributed and needs at least 2 ranks")
    if case.global_sequence_length % case.world_size != 0:
        _reject(reasons, "global sequence must be divisible by the GPU count")
    if case.method == "ring":
        if case.variant not in {"basic", "zigzag"}:
            _reject(reasons, "ring variants are basic and zigzag")
        if case.ulysses_degree != 1 or case.ring_degree != case.world_size:
            _reject(reasons, "pure ring uses ulysses degree 1 and ring degree equal to the world size")
        if case.variant == "zigzag" and case.global_sequence_length % (2 * case.world_size) != 0:
            _reject(reasons, "zigzag ring requires the sequence to be divisible by 2 * world size")
    elif case.method == "distflashattn":
        if case.variant != "balanced":
            _reject(reasons, "DistFlashAttn's required variant is the load-balanced schedule, not ring or Ulysses")
        if case.world_size % 2 != 0:
            _reject(reasons, "audited LightSeq forward schedule is rejected for odd world sizes")
        if case.head_dim not in DISTFLASH_HEAD_DIMS:
            _reject(reasons, f"LightSeq asserts head dim in {sorted(DISTFLASH_HEAD_DIMS)}")
        local = case.global_sequence_length // case.world_size if case.world_size else 0
        if local % DISTFLASH_BLOCK_M != 0:
            _reject(reasons, f"LightSeq BLOCK_M is {DISTFLASH_BLOCK_M}; local sequence {local} is not aligned")
    elif case.method == "ulysses":
        if case.variant != "default":
            _reject(reasons, "Ulysses has no zigzag variant in this matrix")
        if case.ulysses_degree != case.world_size or case.ring_degree != 1:
            _reject(reasons, "pure Ulysses uses ulysses degree equal to the world size and ring degree 1")
        if case.heads % case.world_size != 0:
            _reject(reasons, "attention heads must be divisible by the Ulysses degree")
    elif case.method == "usp":
        if case.variant not in {"basic", "zigzag"}:
            _reject(reasons, "USP variants are basic and zigzag")
        if case.placement not in {"ulysses_low", "ring_low"}:
            _reject(reasons, "USP placement must be ulysses_low or ring_low")
        if case.ulysses_degree is None or case.ring_degree is None:
            _reject(reasons, "USP needs both degrees")
        elif case.ulysses_degree * case.ring_degree != case.world_size:
            _reject(reasons, "Ulysses degree times ring degree must equal the world size")
        elif case.heads % case.ulysses_degree != 0:
            _reject(reasons, "attention heads must be divisible by the Ulysses degree")
        elif case.variant == "zigzag" and case.global_sequence_length % (2 * case.ring_degree) != 0:
            _reject(reasons, "USP zigzag requires the sequence to be divisible by 2 * ring degree")
        elif (
            case.variant == "basic"
            and case.placement in {"ulysses_low", "ring_low"}
            and not basic_usp_preserves_causal_ring_order(case.ulysses_degree, case.ring_degree, case.placement)
        ):
            _reject(
                reasons,
                "basic USP with this placement does not keep contiguous chunks in causal ring order; "
                "upstream basic_extract_local plus ring_low is not a valid causal layout",
            )
        if case.genuine_hybrid != (
            case.ulysses_degree is not None
            and case.ring_degree is not None
            and case.ulysses_degree > 1
            and case.ring_degree > 1
        ):
            _reject(reasons, "genuine_hybrid must be true only when both USP degrees are greater than 1")
    else:
        _reject(reasons, f"unknown method {case.method}")
    return reasons


def _base_fields(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "dtype": config["dtype"],
        "causal": bool(config["causal"]),
        "batch_size": int(config["batch_size"]),
        "heads": int(config["num_heads"]),
        "kv_heads": int(config["kv_heads"]),
        "head_dim": int(config["head_dim"]),
        "hidden_size": int(config["hidden_size"]),
        "dropout": float(config["dropout"]),
    }


def _placement_is_distinct(world_size: int, ulysses_degree: int, ring_degree: int, placement: str) -> bool:
    if placement == "ulysses_low":
        return True
    return not build_group_maps(world_size, ulysses_degree, ring_degree, placement).groups_identical_to(
        build_group_maps(world_size, ulysses_degree, ring_degree, "ulysses_low")
    )


def iter_candidate_cases(config: dict[str, Any]) -> list[Case]:
    common = _base_fields(config)
    phases = list(config["phases"])
    cases: list[Case] = []
    for phase in phases:
        for length in config["sequence_lengths"]:
            cases.append(
                Case(
                    method="single_gpu_sdpa",
                    variant="pytorch_sdpa",
                    phase=phase,
                    topology="single_node",
                    nodes=1,
                    world_size=1,
                    global_sequence_length=int(length),
                    ulysses_degree=None,
                    ring_degree=None,
                    placement=None,
                    genuine_hybrid=False,
                    **common,
                )
            )
            cases.append(
                Case(
                    method="single_gpu_flash",
                    variant="flash_attn_package",
                    phase=phase,
                    topology="single_node",
                    nodes=1,
                    world_size=1,
                    global_sequence_length=int(length),
                    ulysses_degree=None,
                    ring_degree=None,
                    placement=None,
                    genuine_hybrid=False,
                    **common,
                )
            )
    for topology in config["topologies"]:
        for gpu_count in topology["gpu_counts"]:
            for phase in phases:
                for length in config["sequence_lengths"]:
                    shared = {
                        "phase": phase,
                        "topology": topology["name"],
                        "nodes": int(topology["nodes"]),
                        "world_size": int(gpu_count),
                        "global_sequence_length": int(length),
                        **common,
                    }
                    for variant in ("basic", "zigzag"):
                        cases.append(
                            Case(
                                method="ring",
                                variant=variant,
                                ulysses_degree=1,
                                ring_degree=int(gpu_count),
                                placement=None,
                                genuine_hybrid=False,
                                **shared,
                            )
                        )
                    cases.append(
                        Case(
                            method="distflashattn",
                            variant="balanced",
                            ulysses_degree=None,
                            ring_degree=None,
                            placement=None,
                            genuine_hybrid=False,
                            **shared,
                        )
                    )
                    cases.append(
                        Case(
                            method="ulysses",
                            variant="default",
                            ulysses_degree=int(gpu_count),
                            ring_degree=1,
                            placement=None,
                            genuine_hybrid=False,
                            **shared,
                        )
                    )
                    degree_pairs = config["usp_degrees"].get(str(gpu_count), config["usp_degrees"].get(gpu_count, []))
                    for ulysses_degree, ring_degree in degree_pairs:
                        genuine = int(ulysses_degree) > 1 and int(ring_degree) > 1
                        for variant in ("basic", "zigzag"):
                            for placement in ("ulysses_low", "ring_low"):
                                distinct = _placement_is_distinct(
                                    int(gpu_count), int(ulysses_degree), int(ring_degree), placement
                                )
                                cases.append(
                                    Case(
                                        method="usp",
                                        variant=variant,
                                        ulysses_degree=int(ulysses_degree),
                                        ring_degree=int(ring_degree),
                                        placement=placement,
                                        genuine_hybrid=genuine,
                                        placement_distinct=distinct,
                                        **shared,
                                    )
                                )
    return cases


def classify_case(case: Case, hardware: dict[str, Any], dependencies: dict[str, bool]) -> Case:
    reasons = validate_case(case)
    if case.method == "usp" and case.placement == "ring_low" and case.placement_distinct is False and not reasons:
        case.status = "duplicate"
        case.reasons = ["ring_low rank map matches ulysses_low, so this is not a second placement"]
        return case
    if reasons:
        case.status = "rejected"
        case.reasons = reasons
        return case
    blocked: list[str] = []
    if hardware.get("hypothetical"):
        case.hypothetical = True
        case.status = "hypothetical"
        case.reasons = ["planning row for an unobserved GPU allocation; not a measurement"]
        return case
    if case.nodes == 2 and not hardware.get("two_node_100gbe_verified"):
        blocked.append("two-server 100 Gb Ethernet is not verified in this environment")
    visible = int(hardware.get("cuda_device_count") or 0)
    if case.nodes == 1 and visible < case.world_size:
        blocked.append(f"CUDA device count is {visible}; this case needs {case.world_size}")
    dependency_name = {
        "ring": "ring_flash_attn",
        "distflashattn": "lightseq",
        "ulysses": "deepspeed",
        "usp": "yunchang",
        "single_gpu_flash": "flash_attn",
        "single_gpu_sdpa": "torch",
    }[case.method]
    if case.method != "single_gpu_sdpa" and not dependencies.get(dependency_name, False):
        blocked.append(f"upstream dependency {dependency_name} is not importable")
    if case.method == "single_gpu_sdpa" and not hardware.get("cuda_available"):
        blocked.append("single-GPU SDPA timing requires CUDA; CPU attention is correctness-only")
    if case.method == "single_gpu_flash" and hardware.get("cuda_available"):
        capabilities = hardware.get("compute_capabilities") or []
        if capabilities and all(int(str(item).split(".", 1)[0]) < 8 for item in capabilities):
            blocked.append("visible GPU compute capability is below sm_80; official FlashAttention-2 kernels are not claimed here")
    if blocked:
        case.status = "blocked"
        case.reasons = blocked
        return case
    case.status = "planned"
    case.reasons = []
    return case


def expand_matrix(
    config: dict[str, Any],
    hardware: dict[str, Any],
    dependencies: dict[str, bool] | None = None,
) -> list[Case]:
    dependencies = dependencies or {}
    return [classify_case(case, hardware, dependencies) for case in iter_candidate_cases(config)]


def summarize(cases: list[Case]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        counts[case.status] = counts.get(case.status, 0) + 1
    counts["total"] = len(cases)
    return counts


def require_valid_config(config: dict[str, Any]) -> None:
    required = [
        "batch_size",
        "num_heads",
        "kv_heads",
        "head_dim",
        "hidden_size",
        "dtype",
        "dropout",
        "causal",
        "sequence_lengths",
        "phases",
        "topologies",
        "usp_degrees",
        "seed",
        "warmup_iterations",
        "measured_iterations",
        "independent_repeats",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise InvalidConfiguration(f"benchmark config is missing {missing}")
    if int(config["num_heads"]) * int(config["head_dim"]) != int(config["hidden_size"]):
        raise InvalidConfiguration("config hidden_size does not equal num_heads * head_dim")
    if config["dtype"] not in {"bf16", "fp16"}:
        raise InvalidConfiguration("config dtype must be bf16 or fp16")
    for topology in config["topologies"]:
        if topology["nodes"] == 2 and any(int(count) % 2 for count in topology["gpu_counts"]):
            raise InvalidConfiguration("two-node gpu_counts must be even")
        for gpu_count in topology["gpu_counts"]:
            pairs = config["usp_degrees"].get(str(gpu_count), config["usp_degrees"].get(int(gpu_count)))
            if not pairs:
                raise InvalidConfiguration(f"usp_degrees missing for {gpu_count} GPUs")
            for pair in pairs:
                if len(pair) != 2 or int(pair[0]) * int(pair[1]) != int(gpu_count):
                    raise InvalidConfiguration(f"USP degree pair {pair} does not multiply to {gpu_count}")
