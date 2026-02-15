# WX-AFD

Fine-tuning a language model to generate [Area Forecast Discussions](https://forecast.weather.gov/product.php?site=LMK&issuedby=LMK&product=AFD) from structured weather data. Trained on Louisville WFO (LMK) products.

![Python](https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-in%20development-orange)

---

## Pipeline

| Step | Script | Description |
|------|--------|-------------|
| 1 | [`01_scrape_afds.py`](01_scrape_afds.py) | Scrape AFD products from NWS |
| 2 | [`02_fetch_weather.py`](02_fetch_weather.py) | Fetch matching weather observations |
| 3 | [`03_build_dataset.py`](03_build_dataset.py) | Build prompt-completion pairs |
| 4 | [`04_train.ipynb`](04_train.ipynb) | Fine-tune on NCAR Derecho (A100s) |

## Quick Start

```bash
pip install -r requirements.txt
python 01_scrape_afds.py
python 02_fetch_weather.py
python 03_build_dataset.py
```

For HPC setup, see [`setup_derecho.sh`](setup_derecho.sh).

## Docs

- [Product Requirements Document](WX-AFD-PRD.md)
- [Training Config](configs/wx-afd-dora.yml)

## Creator

**ringusTheImp**
