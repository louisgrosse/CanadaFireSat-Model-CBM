# 🔥🛰️ Sparse Autoencoder Concept Discovery for Interpretable Wildfire Forecasting

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)

# CanadaFireSat-Model-CBM

Research code for **interpretable wildfire forecasting** on **CanadaFireSat** using a **Concept Bottleneck Model (CBM)** built from **Sparse Autoencoder (SAE)** concepts.

- 💿 Dataset on [Hugging Face](https://huggingface.co/datasets/EPFL-ECEO/CanadaFireSat) <br>
- 📊 Data repository on [GitHub](https://github.com/eceo-epfl/CanadaFireSat-Data)

The core idea is to forecast **binary wildfire occurrence maps at 100 m resolution** from **multi-spectral Sentinel-2 time series**, while enabling **concept-level explanations and edits**.

---

## Overview

This repo implements a “discover-then-name” concept bottleneck pipeline:

1. **Frozen MS-CLIP encoder** produces patch-level features for each Sentinel-2 observation.
2. A **temporal transformer** aggregates per-patch time series and injects **day-of-year (DOY)** information into attention (Q/K modulation).
3. Features are projected into the **512-D CLIP embedding space** (for text/image cosine alignment).
4. A **Top-K Sparse Autoencoder** is trained on these 512-D patch embeddings to learn an **overcomplete sparse concept space**.
5. A **CBM head** operates on sparse concept activations, enabling:
   - per-concept contribution maps
   - concept ablations / edits
   - automatic concept naming via cosine similarity to text dictionaries (e.g., derived from SSL4EO captions)

---

## Repository structure

- `src/models/MSClipTemporalCBM.py` — MS-CLIP temporal model + CBM wiring (main forecasting model)
- `src/CBM/` — concept bottleneck utilities (concept tensors, naming/alignment helpers, edits/ablations, analysis)

---

## Data

Experiments target the **CanadaFireSat** benchmark:
- inputs: Sentinel-2 multi-spectral time series (typically **1–5 observations across the ~2 months before fire**)
- outputs: **binary wildfire occurrence mask at 100 m** (forecast horizon defined by the dataset setup)

> You must obtain the dataset separately and configure local paths.

---

## Getting started

1. Create a Python environment (conda/venv).
2. Install dependencies (e.g., `pip install -r requirements.txt` if provided).
3. Point configs / scripts to your dataset location.

---

## Typical experiment flow

1. **Baseline**: run the frozen MS-CLIP + temporal aggregation model for segmentation.
2. **SAE training**: train a Top-K SAE on the **512-D** temporally aggregated patch embeddings.
3. **CBM integration**: replace/augment the dense head with the concept bottleneck representation.
4. **Interpretability**:
   - name concepts with cosine similarity to text dictionaries
   - visualize concept activation maps and per-concept contributions
   - run concept ablations/edits and observe changes in predicted risk maps

---

## Notes

This is research code tied to an academic project; expect to adapt paths, configs, and entrypoints to your environment.

---

## Citation

If you use this repository, please cite the associated thesis:

**L. Grosse**, *Sparse Autoencoder Concept Discovery for Interpretable Wildfire Forecasting*, Master Thesis, 2026.
