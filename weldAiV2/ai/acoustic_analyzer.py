"""
Acoustic Integrity Monitoring Module (INNOVATION).
Captures and processes real-time acoustic emissions from welding to detect
defects like porosity, cracks, and lack of fusion.

This is the innovative differentiator: correlating acoustic signatures
with welding parameters for early defect detection BEFORE traditional NDE.
"""

import numpy as np
import joblib
from pathlib import Path
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
from collections import deque

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "acoustic_classifier.joblib"


class AcousticIntegrityAnalyzer:
    """
    Real-time acoustic emission (AE) analysis for weld integrity monitoring.

    Processing pipeline:
    1. Signal acquisition (simulated waveforms)
    2. Feature extraction (time-domain + frequency-domain)
    3. Pattern classification (normal vs. defect types)
    4. Quality index computation
    5. Correlation with welding parameters
    """

    DEFECT_CLASSES = ["normal", "porosity", "crack", "lack_of_fusion", "slag_inclusion"]

    def __init__(self, sample_rate: int = 1000000):
        self.sample_rate = sample_rate
        self.classifier = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_report = {}
        self._feature_names = sorted([
            "amplitude_db", "counts", "duration_ms", "energy", "kurtosis",
            "peak_frequency_hz", "rise_time_ms", "rms_amplitude",
            "signal_strength", "spectral_centroid_hz", "spectral_entropy",
            "spectral_spread_hz", "zero_crossing_rate",
        ])

        self.reading_buffer = deque(maxlen=500)
        self.quality_history = deque(maxlen=1000)
        self.classification_history = deque(maxlen=1000)

        self.defect_signatures = {
            "porosity": {
                "freq_range": (150000, 350000),
                "amplitude_range": (60, 85),
                "energy_range": (400, 1200),
                "burst_pattern": True,
            },
            "crack": {
                "freq_range": (250000, 500000),
                "amplitude_range": (70, 95),
                "energy_range": (800, 2000),
                "burst_pattern": True,
            },
            "lack_of_fusion": {
                "freq_range": (100000, 250000),
                "amplitude_range": (55, 75),
                "energy_range": (300, 800),
                "burst_pattern": False,
            },
            "slag_inclusion": {
                "freq_range": (120000, 280000),
                "amplitude_range": (50, 70),
                "energy_range": (200, 600),
                "burst_pattern": False,
            },
        }

        self._try_load_pretrained()

    def _try_load_pretrained(self):
        """Load pre-trained model from disk if available."""
        if MODEL_PATH.exists():
            try:
                artifact = joblib.load(MODEL_PATH)
                self.classifier = artifact["model"]
                self.scaler = artifact["scaler"]
                self._feature_names = artifact.get("feature_cols", self._feature_names)
                raw = artifact.get("metrics", {})
                per_class = raw.get("per_class_metrics", {})
                report = {}
                for cls_name, cls_m in per_class.items():
                    report[cls_name] = {
                        "precision": cls_m.get("precision", 0),
                        "recall": cls_m.get("recall", 0),
                        "f1-score": cls_m.get("f1", 0),
                        "support": cls_m.get("support", 0),
                    }
                report["accuracy"] = raw.get("test_accuracy", 0)
                report["macro avg"] = {
                    "precision": 0, "recall": 0,
                    "f1-score": raw.get("test_macro_f1", 0), "support": 0,
                }
                report["weighted avg"] = {
                    "precision": 0, "recall": 0,
                    "f1-score": raw.get("test_weighted_f1", 0), "support": 0,
                }
                report["_full"] = raw
                self.training_report = report
                self.is_trained = True
            except Exception:
                pass

    def extract_features(self, waveform: np.ndarray = None,
                         reading: dict = None) -> dict:
        """Extract acoustic features from waveform or reading dict."""
        features = {}

        if waveform is not None and len(waveform) > 0:
            features.update(self._time_domain_features(waveform))
            features.update(self._frequency_domain_features(waveform))
        elif reading:
            if "waveform" in reading and reading["waveform"]:
                waveform = np.array(reading["waveform"])
                features.update(self._time_domain_features(waveform))
                features.update(self._frequency_domain_features(waveform))

        if reading:
            features["amplitude_db"] = reading.get("amplitude_db", 0)
            features["peak_frequency_hz"] = reading.get("peak_frequency_hz", 0)
            features["energy"] = reading.get("energy", 0)
            features["duration_ms"] = reading.get("duration_ms", 0)
            features["rise_time_ms"] = reading.get("rise_time_ms", 0)
            features["counts"] = reading.get("counts", 0)
            features["signal_strength"] = reading.get("signal_strength", 0)
            features.setdefault("spectral_centroid_hz",
                                reading.get("spectral_centroid_hz", 200000))
            features.setdefault("spectral_spread_hz",
                                reading.get("spectral_spread_hz", 50000))

        return features

    def _time_domain_features(self, waveform: np.ndarray) -> dict:
        """Extract time-domain features."""
        abs_wave = np.abs(waveform)
        return {
            "rms_amplitude": float(np.sqrt(np.mean(waveform ** 2))),
            "peak_amplitude": float(np.max(abs_wave)),
            "crest_factor": float(np.max(abs_wave) / (np.sqrt(np.mean(waveform ** 2)) + 1e-10)),
            "kurtosis": float(stats.kurtosis(waveform)),
            "skewness": float(stats.skew(waveform)),
            "zero_crossing_rate": float(
                np.sum(np.diff(np.sign(waveform)) != 0) / len(waveform)
            ),
            "signal_energy": float(np.sum(waveform ** 2)),
            "mean_abs_value": float(np.mean(abs_wave)),
        }

    def _frequency_domain_features(self, waveform: np.ndarray) -> dict:
        """Extract frequency-domain features using FFT."""
        n = len(waveform)
        fft_vals = np.abs(fft(waveform))[:n // 2]
        freqs = fftfreq(n, 1.0 / self.sample_rate)[:n // 2]

        fft_norm = fft_vals / (np.sum(fft_vals) + 1e-10)
        spectral_centroid = np.sum(freqs * fft_norm)
        spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * fft_norm))

        spectral_entropy = -np.sum(fft_norm * np.log2(fft_norm + 1e-10))

        peak_idx = np.argmax(fft_vals)
        peak_freq = freqs[peak_idx] if peak_idx < len(freqs) else 0

        band_edges = [0, 100000, 200000, 300000, 400000, 500000]
        band_energies = {}
        for i in range(len(band_edges) - 1):
            mask = (freqs >= band_edges[i]) & (freqs < band_edges[i + 1])
            band_energies[f"band_{band_edges[i]//1000}k_{band_edges[i+1]//1000}k"] = float(
                np.sum(fft_vals[mask] ** 2)
            )

        features = {
            "spectral_centroid_hz": float(spectral_centroid),
            "spectral_spread_hz": float(spectral_spread),
            "spectral_entropy": float(spectral_entropy),
            "dominant_frequency_hz": float(peak_freq),
            "spectral_flatness": float(
                np.exp(np.mean(np.log(fft_vals + 1e-10))) / (np.mean(fft_vals) + 1e-10)
            ),
        }
        features.update(band_energies)
        return features

    def train_classifier(self, n_samples: int = 6000):
        """Train the acoustic defect classifier with synthetic data."""
        X, y = self._generate_training_data(n_samples)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.classifier.fit(X_train_scaled, y_train)
        self.is_trained = True

        from sklearn.metrics import classification_report
        y_pred = self.classifier.predict(X_test_scaled)
        self.training_report = classification_report(
            y_test, y_pred, target_names=self.DEFECT_CLASSES, output_dict=True
        )

    def _generate_training_data(self, n_samples: int):
        """Generate labeled training data for acoustic classification."""
        try:
            from ai.data_generator import generate_acoustic_dataset
            df = generate_acoustic_dataset(n_samples=n_samples)
            feature_cols = sorted([
                "amplitude_db", "counts", "duration_ms", "energy", "kurtosis",
                "peak_frequency_hz", "rise_time_ms", "rms_amplitude",
                "signal_strength", "spectral_centroid_hz", "spectral_entropy",
                "spectral_spread_hz", "zero_crossing_rate",
            ])
            self._feature_names = feature_cols
            X = df[feature_cols].values
            y = df["defect_class_idx"].values
            return X, y
        except ImportError:
            pass

        np.random.seed(42)
        features_list = []
        labels = []

        class_probs = [0.65, 0.12, 0.08, 0.10, 0.05]

        for _ in range(n_samples):
            cls_idx = np.random.choice(len(self.DEFECT_CLASSES), p=class_probs)
            cls = self.DEFECT_CLASSES[cls_idx]

            if cls == "normal":
                features = {
                    "amplitude_db": np.random.uniform(25, 55),
                    "peak_frequency_hz": np.random.uniform(100000, 350000),
                    "energy": np.random.uniform(50, 400),
                    "duration_ms": np.random.uniform(0.5, 15),
                    "rise_time_ms": np.random.uniform(0.1, 3),
                    "counts": np.random.randint(5, 40),
                    "signal_strength": np.random.uniform(0.1, 0.5),
                    "rms_amplitude": np.random.uniform(0.01, 0.1),
                    "kurtosis": np.random.uniform(2, 5),
                    "spectral_centroid_hz": np.random.uniform(150000, 250000),
                    "spectral_spread_hz": np.random.uniform(30000, 80000),
                    "spectral_entropy": np.random.uniform(5, 10),
                    "zero_crossing_rate": np.random.uniform(0.3, 0.5),
                }
            else:
                sig = self.defect_signatures[cls]
                f_lo, f_hi = sig["freq_range"]
                a_lo, a_hi = sig["amplitude_range"]
                e_lo, e_hi = sig["energy_range"]

                features = {
                    "amplitude_db": np.random.uniform(a_lo, a_hi),
                    "peak_frequency_hz": np.random.uniform(f_lo, f_hi),
                    "energy": np.random.uniform(e_lo, e_hi),
                    "duration_ms": np.random.uniform(5, 50),
                    "rise_time_ms": np.random.uniform(1, 10),
                    "counts": np.random.randint(30, 200),
                    "signal_strength": np.random.uniform(0.4, 1.0),
                    "rms_amplitude": np.random.uniform(0.05, 0.3),
                    "kurtosis": np.random.uniform(4, 12),
                    "spectral_centroid_hz": np.random.uniform(f_lo, f_hi),
                    "spectral_spread_hz": np.random.uniform(50000, 120000),
                    "spectral_entropy": np.random.uniform(3, 7),
                    "zero_crossing_rate": np.random.uniform(0.2, 0.45),
                }

            noise_factor = 0.1
            for k in features:
                if isinstance(features[k], float):
                    features[k] += np.random.normal(0, abs(features[k]) * noise_factor)

            features_list.append(features)
            labels.append(cls_idx)

        feature_names = sorted(features_list[0].keys())
        X = np.array([[f.get(fn, 0) for fn in feature_names] for f in features_list])
        self._feature_names = feature_names

        return X, np.array(labels)

    def classify(self, reading: dict) -> dict:
        """Classify an acoustic reading as normal or defect type."""
        if not self.is_trained:
            self.train_classifier()

        features = self.extract_features(reading=reading)
        feature_vector = [features.get(fn, 0) for fn in self._feature_names]
        X = np.array(feature_vector).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        prediction = self.classifier.predict(X_scaled)[0]
        probabilities = self.classifier.predict_proba(X_scaled)[0]

        class_name = self.DEFECT_CLASSES[prediction]
        confidence = float(max(probabilities))

        result = {
            "classification": class_name,
            "confidence": round(confidence, 4),
            "probabilities": {
                self.DEFECT_CLASSES[i]: round(float(p), 4)
                for i, p in enumerate(probabilities)
            },
            "is_defect": class_name != "normal",
            "features": {k: round(float(v), 4) for k, v in features.items()
                         if isinstance(v, (int, float))},
        }

        self.classification_history.append(result)
        return result

    def compute_quality_index(self, readings: list = None) -> dict:
        """
        Compute acoustic quality index (0-1) from recent readings.
        Higher = better quality.
        """
        if readings is None:
            readings = list(self.reading_buffer)

        if not readings:
            return {"quality_index": 1.0, "status": "no_data"}

        classifications = []
        for r in readings:
            if isinstance(r, dict) and "classification" in r:
                classifications.append(r)
            else:
                cls_result = self.classify(r)
                classifications.append(cls_result)

        normal_count = sum(1 for c in classifications if not c.get("is_defect", False))
        total = len(classifications)
        normal_ratio = normal_count / max(total, 1)

        defect_confidences = [
            c["confidence"] for c in classifications if c.get("is_defect", False)
        ]
        avg_defect_confidence = np.mean(defect_confidences) if defect_confidences else 0

        quality_index = normal_ratio * 0.7 + (1 - avg_defect_confidence) * 0.3

        defect_counts = {}
        for c in classifications:
            cls = c.get("classification", "unknown")
            defect_counts[cls] = defect_counts.get(cls, 0) + 1

        if quality_index > 0.85:
            status = "excellent"
        elif quality_index > 0.7:
            status = "good"
        elif quality_index > 0.5:
            status = "warning"
        else:
            status = "critical"

        result = {
            "quality_index": round(quality_index, 4),
            "status": status,
            "total_readings": total,
            "normal_ratio": round(normal_ratio, 4),
            "defect_distribution": defect_counts,
            "avg_defect_confidence": round(avg_defect_confidence, 4),
        }

        self.quality_history.append(result)
        return result

    def correlate_with_params(self, acoustic_readings: list,
                               welding_params: list) -> dict:
        """
        Correlate acoustic signatures with welding parameters
        to identify parameter-defect relationships.
        """
        if not acoustic_readings or not welding_params:
            return {"correlations": {}, "insights": []}

        min_len = min(len(acoustic_readings), len(welding_params))
        acoustic_readings = acoustic_readings[:min_len]
        welding_params = welding_params[:min_len]

        amplitudes = [r.get("amplitude_db", 0) for r in acoustic_readings]
        energies = [r.get("energy", 0) for r in acoustic_readings]
        currents = [p.get("current_a", 0) for p in welding_params]
        voltages = [p.get("voltage_v", 0) for p in welding_params]
        speeds = [p.get("travel_speed_mm_min", 0) for p in welding_params]

        correlations = {}
        pairs = [
            ("acoustic_amplitude", "current", amplitudes, currents),
            ("acoustic_amplitude", "voltage", amplitudes, voltages),
            ("acoustic_energy", "current", energies, currents),
            ("acoustic_energy", "travel_speed", energies, speeds),
        ]

        for name_a, name_b, data_a, data_b in pairs:
            if len(data_a) > 2 and np.std(data_a) > 0 and np.std(data_b) > 0:
                corr, p_val = stats.pearsonr(data_a, data_b)
                correlations[f"{name_a}_vs_{name_b}"] = {
                    "correlation": round(float(corr), 4),
                    "p_value": round(float(p_val), 6),
                    "significant": p_val < 0.05,
                }

        insights = []
        for pair_name, corr_data in correlations.items():
            if corr_data["significant"]:
                direction = "positive" if corr_data["correlation"] > 0 else "negative"
                strength = "strong" if abs(corr_data["correlation"]) > 0.6 else "moderate"
                insights.append(
                    f"{strength.title()} {direction} correlation between "
                    f"{pair_name.replace('_vs_', ' and ')}: r={corr_data['correlation']}"
                )

        return {
            "correlations": correlations,
            "insights": insights,
            "sample_size": min_len,
        }

    def get_analysis_summary(self) -> dict:
        """Get summary of acoustic analysis."""
        recent_quality = list(self.quality_history)[-10:] if self.quality_history else []
        recent_class = list(self.classification_history)[-20:] if self.classification_history else []

        defect_rate = 0
        if recent_class:
            defect_rate = sum(1 for c in recent_class if c.get("is_defect")) / len(recent_class)

        return {
            "model_trained": self.is_trained,
            "total_classifications": len(self.classification_history),
            "recent_defect_rate": round(defect_rate, 4),
            "recent_quality_scores": [q["quality_index"] for q in recent_quality],
            "defect_classes": self.DEFECT_CLASSES,
            "training_report": self.training_report,
        }
