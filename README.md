# WeldTracker Audio ML Evidence

This repository contains the audio-classification portion of WeldTracker:
dataset generation, model training, held-out evaluation, and acoustic feature
analysis.

## Project evidence

- [Acoustic classifier](weldAiV2/ai/acoustic_analyzer.py)
- [Synthetic dataset generator](weldAiV2/ai/data_generator.py)
- [Complete model training pipeline](weldAiV2/ai/train_all.py)
- [Held-out metrics and confusion matrix](weldAiV2/models/training_report.json)
- [Trained acoustic classifier artifact](weldAiV2/models/acoustic_classifier.joblib)
- [Acoustic monitoring interface](weldAiV2/pages/4_Acoustic_Monitoring.py)

## Existing synthetic-data results

The existing acoustic classifier was trained on 6,000 generated samples and
evaluated with a stratified held-out test set.

| Metric | Result |
|---|---:|
| Test accuracy | 96.33% |
| Macro F1 | 92.35% |
| Weighted F1 | 96.31% |
| Cross-validation macro F1 | 90.66% |

The full per-class metrics and confusion matrix are in
[training_report.json](weldAiV2/models/training_report.json).

## Reproduce model training

```powershell
cd weldAiV2
python -m pip install -r requirements.txt
python ai/train_all.py
```

The pipeline generates the synthetic datasets, trains the comparison models,
performs 5-fold stratified cross-validation and randomized hyperparameter
tuning, and writes the saved classifier artifacts.

## Scope

These synthetic-data results demonstrate the Python training and evaluation
pipeline. They are not evidence of real-world welding performance or an
existing embedded deployment.
