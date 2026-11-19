"""
Centralized Synthetic Data Generator for WeldTracker AI Models.
Generates physics-based, realistic welding datasets for:
  1. Anomaly detection (sensor-level readings)
  2. Defect prediction (weld-level aggregated features)
  3. Acoustic classification (AE signal features)

All distributions are calibrated against real-world welding parameter ranges
from ASME IX / ISO 15614 standards and S2A INDUSTRIE field data.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

WELDING_PROCESSES = {
    "SMAW": {
        "current_a": (90, 180, 135, 15),
        "voltage_v": (20, 28, 24, 1.5),
        "travel_speed_mm_min": (80, 250, 165, 25),
        "interpass_temp_c": (20, 250, 130, 35),
    },
    "GTAW": {
        "current_a": (60, 250, 150, 20),
        "voltage_v": (10, 20, 14, 2),
        "travel_speed_mm_min": (50, 200, 110, 20),
        "interpass_temp_c": (20, 200, 100, 30),
    },
    "FCAW": {
        "current_a": (150, 350, 250, 30),
        "voltage_v": (24, 34, 28, 2),
        "travel_speed_mm_min": (150, 400, 250, 35),
        "interpass_temp_c": (20, 250, 140, 35),
    },
}


def _clamp(val, lo, hi):
    return max(lo, min(hi, val))


def generate_anomaly_dataset(n_samples: int = 5000, anomaly_ratio: float = 0.08,
                             seed: int = 42) -> pd.DataFrame:
    """
    Generate sensor-level readings for anomaly detection training.
    ~92 % normal, ~8 % anomalous (threshold, drift, spike types).
    """
    rng = np.random.RandomState(seed)
    rows = []

    n_anomaly = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomaly

    for _ in range(n_normal):
        proc = rng.choice(list(WELDING_PROCESSES.keys()), p=[0.5, 0.2, 0.3])
        spec = WELDING_PROCESSES[proc]
        current = _clamp(rng.normal(spec["current_a"][2], spec["current_a"][3]),
                         spec["current_a"][0], spec["current_a"][1])
        voltage = _clamp(rng.normal(spec["voltage_v"][2], spec["voltage_v"][3]),
                         spec["voltage_v"][0], spec["voltage_v"][1])
        speed = _clamp(rng.normal(spec["travel_speed_mm_min"][2], spec["travel_speed_mm_min"][3]),
                       spec["travel_speed_mm_min"][0], spec["travel_speed_mm_min"][1])
        temp = _clamp(rng.normal(spec["interpass_temp_c"][2], spec["interpass_temp_c"][3]),
                      spec["interpass_temp_c"][0], spec["interpass_temp_c"][1])
        heat_input = (current * voltage * 60) / (speed * 1000)

        rows.append({
            "current_a": round(current, 2),
            "voltage_v": round(voltage, 2),
            "travel_speed_mm_min": round(speed, 2),
            "interpass_temp_c": round(temp, 2),
            "heat_input_kj_mm": round(heat_input, 4),
            "process": proc,
            "is_anomaly": 0,
            "anomaly_type": "none",
        })

    anomaly_types = ["threshold_breach", "spike", "drift_high", "drift_low", "multi_param"]
    for _ in range(n_anomaly):
        a_type = rng.choice(anomaly_types)
        proc = rng.choice(list(WELDING_PROCESSES.keys()), p=[0.5, 0.2, 0.3])
        spec = WELDING_PROCESSES[proc]

        current = rng.normal(spec["current_a"][2], spec["current_a"][3])
        voltage = rng.normal(spec["voltage_v"][2], spec["voltage_v"][3])
        speed = rng.normal(spec["travel_speed_mm_min"][2], spec["travel_speed_mm_min"][3])
        temp = rng.normal(spec["interpass_temp_c"][2], spec["interpass_temp_c"][3])

        if a_type == "threshold_breach":
            param = rng.choice(["current", "voltage", "speed", "temp"])
            if param == "current":
                current = rng.choice([rng.uniform(50, spec["current_a"][0] - 10),
                                      rng.uniform(spec["current_a"][1] + 10, 300)])
            elif param == "voltage":
                voltage = rng.choice([rng.uniform(8, spec["voltage_v"][0] - 2),
                                      rng.uniform(spec["voltage_v"][1] + 2, 45)])
            elif param == "speed":
                speed = rng.choice([rng.uniform(30, spec["travel_speed_mm_min"][0] - 10),
                                    rng.uniform(spec["travel_speed_mm_min"][1] + 20, 500)])
            else:
                temp = rng.uniform(spec["interpass_temp_c"][1] + 10, 350)
        elif a_type == "spike":
            current += rng.choice([-1, 1]) * rng.uniform(40, 80)
            voltage += rng.choice([-1, 1]) * rng.uniform(5, 12)
        elif a_type == "drift_high":
            current += rng.uniform(20, 50)
            voltage += rng.uniform(3, 8)
        elif a_type == "drift_low":
            current -= rng.uniform(20, 50)
            speed -= rng.uniform(30, 80)
        elif a_type == "multi_param":
            current += rng.choice([-1, 1]) * rng.uniform(30, 60)
            voltage += rng.choice([-1, 1]) * rng.uniform(4, 10)
            speed += rng.choice([-1, 1]) * rng.uniform(40, 100)

        heat_input = (current * voltage * 60) / (max(speed, 10) * 1000)
        rows.append({
            "current_a": round(current, 2),
            "voltage_v": round(voltage, 2),
            "travel_speed_mm_min": round(max(speed, 10), 2),
            "interpass_temp_c": round(max(temp, 0), 2),
            "heat_input_kj_mm": round(heat_input, 4),
            "process": proc,
            "is_anomaly": 1,
            "anomaly_type": a_type,
        })

    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def generate_defect_dataset(n_samples: int = 5000, defect_ratio: float = 0.15,
                            seed: int = 42) -> pd.DataFrame:
    """
    Generate weld-level aggregated features for defect prediction.
    Each row represents one completed weld joint with pass-level stats.
    """
    rng = np.random.RandomState(seed)
    rows = []

    n_defect = int(n_samples * defect_ratio)
    n_normal = n_samples - n_defect

    def _make_row(is_defect: bool, severity: str) -> dict:
        if is_defect:
            current_mean = rng.normal(140, 25)
            current_std = rng.uniform(6, 22)
            voltage_mean = rng.normal(24, 4)
            voltage_std = rng.uniform(1.5, 5.5)
            speed_mean = rng.normal(155, 45)
            speed_std = rng.uniform(8, 25)
            temp_mean = rng.uniform(80, 220)
            temp_max = temp_mean + rng.uniform(20, 80)
            acoustic_energy = rng.uniform(300, 1500)
            acoustic_amp = rng.uniform(45, 90)
            acoustic_anomaly = rng.uniform(0.05, 0.50)
            deviation_score = rng.uniform(0.2, 0.9)
        else:
            current_mean = rng.normal(135, 12)
            current_std = rng.uniform(1.5, 10)
            voltage_mean = rng.normal(24, 2)
            voltage_std = rng.uniform(0.3, 2.5)
            speed_mean = rng.normal(165, 25)
            speed_std = rng.uniform(2, 8)
            temp_mean = rng.uniform(80, 180)
            temp_max = temp_mean + rng.uniform(10, 50)
            acoustic_energy = rng.uniform(40, 450)
            acoustic_amp = rng.uniform(20, 55)
            acoustic_anomaly = rng.uniform(0.0, 0.10)
            deviation_score = rng.uniform(0.0, 0.25)

        heat_input = (current_mean * voltage_mean * 60) / (max(speed_mean, 10) * 1000)
        arc_stability = 1.0 / (1.0 + current_std * voltage_std / 10)
        pass_count = rng.randint(3, 12)

        return {
            "current_mean": round(current_mean, 2),
            "current_std": round(current_std, 3),
            "current_min": round(current_mean - rng.uniform(10, 35), 2),
            "current_max": round(current_mean + rng.uniform(10, 35), 2),
            "voltage_mean": round(voltage_mean, 2),
            "voltage_std": round(voltage_std, 3),
            "voltage_min": round(voltage_mean - rng.uniform(1, 5), 2),
            "voltage_max": round(voltage_mean + rng.uniform(1, 5), 2),
            "travel_speed_mean": round(speed_mean, 2),
            "travel_speed_std": round(speed_std, 3),
            "interpass_temp_mean": round(temp_mean, 2),
            "interpass_temp_max": round(temp_max, 2),
            "heat_input_mean": round(heat_input, 4),
            "heat_input_std": round(rng.uniform(0.03, 0.35 if is_defect else 0.12), 4),
            "current_voltage_ratio": round(current_mean / max(voltage_mean, 1), 4),
            "arc_stability_index": round(arc_stability, 4),
            "param_deviation_score": round(deviation_score, 4),
            "pass_count": pass_count,
            "acoustic_energy_mean": round(acoustic_energy, 2),
            "acoustic_amplitude_max": round(acoustic_amp, 2),
            "acoustic_anomaly_ratio": round(acoustic_anomaly, 4),
            "has_defect": 1 if is_defect else 0,
            "defect_severity": severity,
        }

    for _ in range(n_normal):
        rows.append(_make_row(False, "none"))

    for _ in range(n_defect):
        sev = rng.choice(["minor", "major"], p=[0.6, 0.4])
        rows.append(_make_row(True, sev))

    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def generate_acoustic_dataset(n_samples: int = 6000, seed: int = 42) -> pd.DataFrame:
    """
    Generate acoustic emission features for 5-class classification.
    Classes: normal, porosity, crack, lack_of_fusion, slag_inclusion.
    """
    rng = np.random.RandomState(seed)

    DEFECT_CLASSES = ["normal", "porosity", "crack", "lack_of_fusion", "slag_inclusion"]
    class_probs = [0.60, 0.13, 0.09, 0.11, 0.07]

    defect_signatures = {
        "porosity": {
            "freq_range": (150000, 350000), "amplitude_range": (60, 85),
            "energy_range": (400, 1200), "duration_range": (5, 40),
            "rise_time_range": (1, 8), "counts_range": (30, 180),
            "signal_strength_range": (0.4, 0.9), "rms_range": (0.05, 0.25),
            "kurtosis_range": (4, 10), "zcr_range": (0.2, 0.42),
            "entropy_range": (3.5, 7),
        },
        "crack": {
            "freq_range": (250000, 500000), "amplitude_range": (70, 95),
            "energy_range": (800, 2000), "duration_range": (8, 55),
            "rise_time_range": (2, 12), "counts_range": (50, 220),
            "signal_strength_range": (0.5, 1.0), "rms_range": (0.08, 0.35),
            "kurtosis_range": (5, 14), "zcr_range": (0.18, 0.40),
            "entropy_range": (2.5, 6),
        },
        "lack_of_fusion": {
            "freq_range": (100000, 250000), "amplitude_range": (55, 75),
            "energy_range": (300, 800), "duration_range": (10, 45),
            "rise_time_range": (3, 10), "counts_range": (25, 150),
            "signal_strength_range": (0.35, 0.80), "rms_range": (0.04, 0.20),
            "kurtosis_range": (3, 9), "zcr_range": (0.22, 0.44),
            "entropy_range": (4, 8),
        },
        "slag_inclusion": {
            "freq_range": (120000, 280000), "amplitude_range": (50, 70),
            "energy_range": (200, 600), "duration_range": (8, 35),
            "rise_time_range": (2, 8), "counts_range": (20, 130),
            "signal_strength_range": (0.30, 0.70), "rms_range": (0.03, 0.18),
            "kurtosis_range": (3, 8), "zcr_range": (0.25, 0.48),
            "entropy_range": (4.5, 8.5),
        },
    }

    rows = []
    for _ in range(n_samples):
        cls_idx = rng.choice(len(DEFECT_CLASSES), p=class_probs)
        cls = DEFECT_CLASSES[cls_idx]

        if cls == "normal":
            feat = {
                "amplitude_db": rng.uniform(25, 55),
                "peak_frequency_hz": rng.uniform(100000, 350000),
                "energy": rng.uniform(40, 400),
                "duration_ms": rng.uniform(0.5, 15),
                "rise_time_ms": rng.uniform(0.1, 3),
                "counts": float(rng.randint(5, 40)),
                "signal_strength": rng.uniform(0.08, 0.48),
                "rms_amplitude": rng.uniform(0.005, 0.10),
                "kurtosis": rng.uniform(1.5, 5),
                "spectral_centroid_hz": rng.uniform(140000, 260000),
                "spectral_spread_hz": rng.uniform(25000, 85000),
                "spectral_entropy": rng.uniform(5, 11),
                "zero_crossing_rate": rng.uniform(0.30, 0.52),
            }
        else:
            sig = defect_signatures[cls]
            feat = {
                "amplitude_db": rng.uniform(*sig["amplitude_range"]),
                "peak_frequency_hz": rng.uniform(*sig["freq_range"]),
                "energy": rng.uniform(*sig["energy_range"]),
                "duration_ms": rng.uniform(*sig["duration_range"]),
                "rise_time_ms": rng.uniform(*sig["rise_time_range"]),
                "counts": float(rng.randint(*sig["counts_range"])),
                "signal_strength": rng.uniform(*sig["signal_strength_range"]),
                "rms_amplitude": rng.uniform(*sig["rms_range"]),
                "kurtosis": rng.uniform(*sig["kurtosis_range"]),
                "spectral_centroid_hz": rng.uniform(*sig["freq_range"]),
                "spectral_spread_hz": rng.uniform(45000, 125000),
                "spectral_entropy": rng.uniform(*sig["entropy_range"]),
                "zero_crossing_rate": rng.uniform(*sig["zcr_range"]),
            }

        noise = 0.08
        for k, v in feat.items():
            if isinstance(v, float):
                feat[k] = v + rng.normal(0, abs(v) * noise)

        feat["defect_class"] = cls
        feat["defect_class_idx"] = cls_idx
        rows.append(feat)

    return pd.DataFrame(rows)


def generate_and_save_all(overwrite: bool = False):
    """Generate all datasets and save to data/ directory as CSV files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    files = {
        "anomaly_dataset.csv": lambda: generate_anomaly_dataset(n_samples=5000),
        "defect_dataset.csv": lambda: generate_defect_dataset(n_samples=5000),
        "acoustic_dataset.csv": lambda: generate_acoustic_dataset(n_samples=6000),
    }

    generated = {}
    for fname, gen_fn in files.items():
        fpath = DATA_DIR / fname
        if fpath.exists() and not overwrite:
            print(f"  [SKIP] {fname} already exists")
            generated[fname] = pd.read_csv(fpath)
        else:
            df = gen_fn()
            df.to_csv(fpath, index=False)
            print(f"  [SAVED] {fname}  ({len(df)} rows, {len(df.columns)} cols)")
            generated[fname] = df

    return generated


if __name__ == "__main__":
    print("=" * 60)
    print("WeldTracker AI - Synthetic Dataset Generation")
    print("=" * 60)
    datasets = generate_and_save_all(overwrite=True)
    for name, df in datasets.items():
        print(f"\n--- {name} ---")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        if "is_anomaly" in df.columns:
            print(f"  Anomaly ratio: {df['is_anomaly'].mean():.2%}")
        if "has_defect" in df.columns:
            print(f"  Defect ratio: {df['has_defect'].mean():.2%}")
        if "defect_class" in df.columns:
            print(f"  Class distribution:\n{df['defect_class'].value_counts().to_string()}")
    print("\nDone.")
