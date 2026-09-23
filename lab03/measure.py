from __future__ import annotations

import re
import statistics
from typing import Any

from bench import Bench, measured, read_first, read_text, unknown

import json

# A sample is still warm-up while it exceeds the settled rate by this fraction.
WARMUP_TOL = 0.5

# How many samples must sit strictly above a quantile before that quantile is an
# estimate rather than "the biggest number we saw, wearing a hat".
MIN_SAMPLES_ABOVE = 5

# Percentiles the record carries, in the order the schema lists them.
PERCENTILES = (50, 95, 99)

# The widest gap between neighbouring measurements, as a multiple of the typical
# gap, beyond which the sample is treated as coming from two populations.
MULTIMODAL_GAP_RATIO = 20.0

# Neither side of that gap is a mode unless it holds at least this fraction.
MIN_MODE_FRACTION = 0.10

# Below this many retained samples, modality is not a question worth answering.
MIN_SAMPLES_FOR_MODALITY = 20

# How far the last third of a run may drift from the first third, relative to
# the run's own median, before the run is not one population either.
STATIONARITY_TOL = 0.10
MIN_SAMPLES_FOR_STATIONARITY = 12

THERMAL_ZONES = "sys/devices/virtual/thermal"

POWER_RAIL_CANDIDATES = (
    "sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3/in1_input",
    "sys/bus/i2c/drivers/ina3221/1-0040/iio:device0/in_power0_input",
    "sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",
)

GPU_LOAD_CANDIDATES = (
    "sys/devices/platform/gpu.0/load",
    "sys/devices/gpu.0/load",
)

CPUFREQ_MIN = "sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq"
CPUFREQ_MAX = "sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"


# ===========================================================================
# 1. The loop
# ===========================================================================
def run_timed_iterations(bench: Bench, repeats: int = 100) -> list[float]:
    bench.workload.synchronize()
    times: list[float] = []

    for _ in range(repeats):
        start_time = bench.clock()
        bench.workload.run()
        bench.workload.synchronize()
        end_time = bench.clock()

        elapsed = end_time - start_time
        times.append(elapsed / 1_000_000.0)

    return times


def find_warmup_boundary(samples: list[float]) -> dict[str, Any]:
    if len(samples) < 4:
        return unknown("find_warmup_boundary", "Too few samples")

    settled_median = statistics.median(samples[len(samples) // 2 :])
    if settled_median <= 0:
        return unknown("find_warmup_boundary", "second-half median is not positive")

    threshold = settled_median * (1 + WARMUP_TOL)

    count = 0
    for sample in samples:
        if sample <= threshold:
            break
        count += 1

    return measured(
        count,
        "leading prefix above (1 + 0.5) x median of the run's second half",
        settled_rate_ms=round(settled_median, 4),
        threshold_ms=round(threshold, 4),
        tolerance=WARMUP_TOL,
        retained=len(samples) - count,
    )


def summarize(samples: list[float]) -> dict[str, Any]:
    metric_keys = ("mean", "std", "min", "max", "p50", "p95", "p99")
    if not samples:
        return {"n": 0, **{key: None for key in metric_keys}}

    ordered = sorted(samples)
    n = len(ordered)

    def percentile(q: int) -> float:
        h = (n - 1) * q / 100
        lower = int(h)
        fraction = h - lower
        if lower == n - 1:
            return ordered[lower]
        return ordered[lower] + fraction * (ordered[lower + 1] - ordered[lower])

    values = {
        "n": n,
        "mean": statistics.fmean(ordered),
        "std": statistics.stdev(ordered) if n > 2 else 0.0,
        "min": ordered[0],
        "max": ordered[-1],
        "p50": percentile(50),
        "p95": percentile(95),
        "p99": percentile(99),
    }
    return {
        key: round(value, 4) if isinstance(value, float) else value
        for key, value in values.items()
    }

def is_multimodal(samples: list[float]) -> dict[str, Any]:
    source = (
        "widest trimmed gap >= 20.0x the median gap, with >= 10% of samples on each side"
    )
    if len(samples) < MIN_SAMPLES_FOR_MODALITY:
        return unknown(source, "not enough samples to assess multimodality")

    ordered = sorted(samples)
    trim = int(len(ordered) * 0.05)
    trimmed = ordered[trim : len(ordered) - trim] if trim else ordered
    gaps = [right - left for left, right in zip(trimmed, trimmed[1:])]
    if not gaps:
        return unknown(source, "not enough adjacent samples after trimming")
    typical_gap = statistics.median(gaps)
    if typical_gap <= 0:
        return unknown(source, "timer resolution is too coarse; median gap is not positive")

    widest_gap = max(gaps)
    gap_index = gaps.index(widest_gap)
    ratio = widest_gap / typical_gap

    split_index = trim + gap_index + 1
    left, right = ordered[:split_index], ordered[split_index:]
    left_share = len(left) / len(ordered)
    right_share = len(right) / len(ordered)
    is_split = (
        ratio >= MULTIMODAL_GAP_RATIO
        and left_share >= MIN_MODE_FRACTION
        and right_share >= MIN_MODE_FRACTION
    )
    return measured(
        is_split,
        source,
        gap_ratio=round(ratio, 2),
        widest_gap_ms=round(widest_gap, 5),
        typical_gap_ms=round(typical_gap, 5),
        modes=[
            {
                "n": len(group),
                "share": round(len(group) / len(ordered), 2),
                "median_ms": round(statistics.median(group), 4),
            }
            for group in (left, right)
        ],
    )


def is_stationary(samples: list[float]) -> dict[str, Any]:
    source = "first-third vs last-third median drift <= 10% of overall median"
    if len(samples) < MIN_SAMPLES_FOR_STATIONARITY:
        return unknown(source, "too few samples to divide into thirds")

    overall_median = statistics.median(samples)
    if overall_median <= 0:
        return unknown(source, "overall median is not positive")

    third_size = len(samples) // 3
    first_median = statistics.median(samples[:third_size])
    last_median = statistics.median(samples[-third_size:])
    drift = last_median - first_median
    relative_drift = abs(drift) / overall_median
    direction = "slower" if drift > 0 else "faster" if drift < 0 else "flat"

    return measured(
        relative_drift <= STATIONARITY_TOL,
        source,
        first_third_median_ms=round(first_median, 4),
        last_third_median_ms=round(last_median, 4),
        drift_ms=round(drift, 4),
        drift_relative=round(relative_drift, 4),
        direction=direction,
        tolerance=STATIONARITY_TOL,
    )

# ===========================================================================
# 7. The clock ceiling the run happened under
# ===========================================================================


def probe_power_state(bench: Bench) -> dict[str, Any]:
    source = "nvpmodel -q"
    result = bench.runner(["nvpmodel", "-q"])
    if not result.ok or result.returncode != 0:
        reason = result.error or f"command exited with status {result.returncode}"
        return unknown(source, f"could not query power mode: {reason}")

    lines = result.stdout.splitlines()
    mode_index: int | None = None
    mode_name: str | None = None
    for index, line in enumerate(lines):
        if "NV Power Mode:" not in line:
            continue
        mode_name = line.split("NV Power Mode:", 1)[1].strip() or None
        if index + 1 < len(lines):
            id_match = re.search(r"(?:ID\s*=\s*)?(\d+)", lines[index + 1], re.IGNORECASE)
            if id_match:
                mode_index = int(id_match.group(1))
        break
    if mode_name is None or mode_index is None:
        return unknown(source, "nvpmodel output did not contain a power mode name and following mode ID")

    minimum = read_text(bench.telemetry, CPUFREQ_MIN)
    maximum = read_text(bench.telemetry, CPUFREQ_MAX)
    cpu_source = f"{CPUFREQ_MIN}=... vs {CPUFREQ_MAX}=..."
    if minimum is None or maximum is None:
        clocks = None
        clocks_source = unknown(cpu_source, "could not read one or both CPU frequency limits")
    else:
        try:
            minimum_value, maximum_value = int(minimum), int(maximum)
        except ValueError:
            clocks = None
            clocks_source = unknown(cpu_source, "CPU frequency limit was not an integer")
        else:
            clocks = minimum_value == maximum_value
            clocks_source = measured(
                f"scaling_min_freq={minimum}, scaling_max_freq={maximum}",
                f"{CPUFREQ_MIN} vs {CPUFREQ_MAX}",
            )

    return measured(
        mode_name,
        source,
        mode_index=mode_index,
        jetson_clocks=clocks,
        jetson_clocks_source=clocks_source,
    )



def probe_telemetry(bench: Bench) -> dict[str, Any]:
    thermal_root = bench.telemetry / THERMAL_ZONES
    temperatures: list[tuple[float, str]] = []
    try:
        zones = sorted(thermal_root.iterdir())
    except OSError:
        zones = []
    for zone in zones:
        if not zone.name.startswith("thermal_zone"):
            continue
        raw = read_text(bench.telemetry, f"{THERMAL_ZONES}/{zone.name}/temp")
        if raw is None:
            continue
        try:
            raw_temperature = int(raw)
        except ValueError:
            continue
        if raw_temperature > -1000:
            temperature = raw_temperature / 1000.0
            zone_type = read_text(bench.telemetry, f"{THERMAL_ZONES}/{zone.name}/type")
            temperatures.append((temperature, zone_type or zone.name))

    if temperatures:
        hottest, hottest_zone = max(temperatures, key=lambda item: item[0])
        temperature_record = measured(
            round(hottest, 2),
            f"{THERMAL_ZONES}/*/temp",
            zone=hottest_zone,
            zones_read=len(temperatures),
        )
    else:
        temperature_record = unknown(
            f"{THERMAL_ZONES}/*/temp", "no readable thermal zones reported a valid temperature"
        )

    power_paths = " | ".join(POWER_RAIL_CANDIDATES)
    power_reading = read_first(bench.telemetry, POWER_RAIL_CANDIDATES)
    if power_reading is None:
        power_record = unknown(power_paths, "none of the documented INA3221 rail paths could be read")
    else:
        power_path, raw_power = power_reading
        try:
            power_record = measured(int(raw_power), power_path)
        except ValueError:
            power_record = unknown(power_path, "INA3221 power reading was not an integer")

    gpu_paths = " | ".join(GPU_LOAD_CANDIDATES)
    gpu_reading = read_first(bench.telemetry, GPU_LOAD_CANDIDATES)
    if gpu_reading is None:
        gpu_record = unknown(gpu_paths, "none of the documented GPU load paths could be read")
    else:
        gpu_path, raw_load = gpu_reading
        try:
            gpu_record = measured(int(raw_load) / 10.0, gpu_path, units="per-mille / 10")
        except ValueError:
            gpu_record = unknown(gpu_path, "GPU load reading was not an integer")

    return {
        "temperature_c": temperature_record,
        "power_mw": power_record,
        "gpu_utilization_percent": gpu_record,
    }

## for debugging - uncomment the following lines for debugging.
# if __name__ == "__main__":
    # env = Bench.real()
    # out = find_warmup_boundary(samples)
    # print(out)

# for generating system_report.json
if __name__ == "__main__":
    # calling base environment
    env = Bench.real()

    # get your samples
    samples = run_timed_iterations(env, repeats=100)

    # testing measurments and probes
    report = {
        "warmup_boundary": find_warmup_boundary(samples),
        "summarize_setup": summarize(samples),
        "is_multimodal": is_multimodal(samples),
        "is_stationary": is_stationary(samples),
        "probe_power_state": probe_power_state(env),
        "probe_telemetry": probe_telemetry(env),
    }

    # save samples
    path = "samples_analysis.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4)

    # save report
    path = "system_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
