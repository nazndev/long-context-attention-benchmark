"""Metric definitions from the project specification."""

from __future__ import annotations


def global_tokens_per_second(batch_size: int, global_sequence_length: int, max_rank_step_seconds: float | None) -> float | None:
    if max_rank_step_seconds is None or max_rank_step_seconds <= 0:
        return None
    return batch_size * global_sequence_length / max_rank_step_seconds


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def median(values: list[float]) -> float | None:
    return percentile(values, 0.5)


def p95(values: list[float]) -> float | None:
    return percentile(values, 0.95)


def _union(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return []
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start > last_end:
            merged.append((start, end))
        else:
            merged[-1] = (last_start, max(last_end, end))
    return merged


def _duration(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


def _intersection(left: list[tuple[float, float]], right: list[tuple[float, float]]) -> list[tuple[float, float]]:
    result = []
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end > start:
            result.append((start, end))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return result


def union_duration_ms(intervals: list[tuple[float, float]] | None) -> float | None:
    if intervals is None:
        return None
    duration = _duration(_union(intervals))
    if duration == 0:
        return None
    return duration


def overlap_fraction(
    compute_intervals_ms: list[tuple[float, float]] | None,
    communication_intervals_ms: list[tuple[float, float]] | None,
) -> tuple[float | None, float | None]:
    """Return (overlap_ms, overlap_fraction).

    Fraction is the duration of the intersection of the unions divided by the
    duration of the communication union. It is null when communication duration
    is zero. Async API calls are not intervals and must not be passed here.
    """

    if communication_intervals_ms is None or compute_intervals_ms is None:
        return None, None
    communication = _union(communication_intervals_ms)
    communication_ms = _duration(communication)
    if communication_ms == 0:
        return None, None
    compute = _union(compute_intervals_ms)
    overlap_ms = _duration(_intersection(compute, communication))
    return overlap_ms, overlap_ms / communication_ms


def scaling_efficiency(
    throughput_p: float | None,
    throughput_reference: float | None,
    gpu_count: int,
    reference_gpu_count: int,
) -> float | None:
    """throughput_P / (throughput_reference * P / reference_gpu_count), fixed global length."""

    if (
        throughput_p is None
        or throughput_reference is None
        or throughput_reference == 0
        or gpu_count < 1
        or reference_gpu_count < 1
    ):
        return None
    return throughput_p / (throughput_reference * gpu_count / reference_gpu_count)


def mib_from_bytes(num_bytes: int | None) -> float | None:
    if num_bytes is None:
        return None
    return num_bytes / (1024 * 1024)
