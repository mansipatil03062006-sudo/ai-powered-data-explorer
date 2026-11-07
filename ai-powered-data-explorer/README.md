# AI-Powered Data Explorer

An interactive data science dashboard built with Streamlit. Upload a CSV file to inspect data, assess quality, create charts, and discover statistical patterns.

## Features

- CSV upload and searchable data preview
- Dataset metrics: row/column count, missing cells, duplicate rows
- Statistical summary and column profile
- Missing-value and duplicate analysis
- Interactive histogram, bar, scatter, and box plots
- Correlation heatmap
- Automated insights: missingness, skew, strong correlations
- Isolation Forest anomaly detection
- Download deduplicated CSV
- Included sample sales dataset

> This starter app uses local statistical/ML methods. It does not call a paid AI API.

## Quick start (Windows PowerShell)

1. Install Python 3.10 or newer from https://www.python.org/downloads/ (enable **Add Python to PATH**).
2. Extract this ZIP.
3. Open PowerShell in the extracted `ai-powered-data-explorer` folder.
4. Run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run this once in the same terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

Open the local URL Streamlit prints (usually http://localhost:8501).

## Quick start (macOS/Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Sample data

Click **Load sample sales dataset** on the app's landing page to try the included dataset without uploading a file.

## GitHub

```bash
git init
git add .
git commit -m "Build AI-powered data explorer"
```

Create a public repository on GitHub, then add its remote and push:

```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ai-powered-data-explorer.git
git push -u origin main
```

## Project structure

```text
ai-powered-data-explorer/
├── app.py
├── requirements.txt
├── README.md
└── data/
    └── sample_sales.csv
```

## Limitations

- CSV files only (Excel support can be added).
- Large files may take time to load and render.
- Anomaly detection is a heuristic and should not be treated as a definitive fraud or risk decision.
