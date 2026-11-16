# CarApp – Road Collision Severity Analysis (Great Britain, 2025)

CarApp is a Streamlit dashboard built for a UWS Advanced Data Science coursework project. It analyses the **2025 Department for Transport (DfT) collision dataset** and demonstrates a **Decision Tree Classifier** that groups police-recorded injury collisions as **Slight** or **Serious/Fatal**.

## What's included

- **Dataset overview:** totals for fatal, serious and slight collisions; dataset preview
- **Collision analysis:** interactive charts and tables for weather, lighting, road surface, speed limits, road type and urban/rural areas
- **Data preprocessing:** missing-value checks, exact duplicates and notes about DfT numerical codes
- **Decision Tree:** model training, test accuracy, precision/recall/F1-score, confusion matrix and simplified tree diagram

## Dataset

Source: [DfT Road Safety Open Data](https://www.gov.uk/government/statistical-data-sets/road-safety-open-data)

File expected: `dft-road-casualty-statistics-collision-2025.csv`

The original analysis uses **101,525 collisions** and **44 variables**. Because the full dataset is not included in this small application package, download it from the DfT page or use the CSV supplied in the university ZIP submission. Upload it through the CarApp sidebar after starting the app. **The CSV is not uploaded to GitHub by the app.**

## Run CarApp locally

1. Install Python 3.10 or newer.
2. Download or clone these files and open a terminal in the folder containing `CarApp.py`.
3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   python -m streamlit run CarApp.py
   ```

5. Open the local URL shown in the terminal (normally `http://localhost:8501`).
6. Upload the DfT 2025 collision CSV in the left sidebar.
7. Open the **Decision Tree** tab to view training/testing results.

### Run in Google Colab for a temporary presentation link

Upload `CarApp.py` and the CSV into Colab's `/content` directory, then run:

```python
!pip -q install streamlit pandas numpy matplotlib plotly scikit-learn
```

```python
import subprocess, sys
process = subprocess.Popen([
    sys.executable, '-m', 'streamlit', 'run', '/content/CarApp.py',
    '--server.address=127.0.0.1', '--server.port=8501', '--server.headless=true'
])
```

For an optional public demonstration using a **temporary** Cloudflare tunnel:

```python
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /content/cloudflared
!chmod +x /content/cloudflared
!/content/cloudflared tunnel --url http://127.0.0.1:8501
```

The tunnel address is temporary and changes when the session ends. The tunnel command continues running while the site is available.

## Decision Tree methodology

- Target groups: `0 = Slight`; `1 = Serious/Fatal`.
- Inputs: speed limit, road type, lighting conditions, weather conditions, road surface conditions and urban/rural area.
- Categorical codes use `OneHotEncoder(handle_unknown='ignore')`; speed limit remains numerical.
- Stratified 80/20 train/test split; random seed 42.
- Decision Tree settings: `max_depth=5`, `min_samples_leaf=100`, `class_weight='balanced'`, `random_state=42`.
- Encoding is fitted on **training data only**, inside a scikit-learn pipeline.
- With the original 2025 CSV, the accompanying notebook obtained **58.12% accuracy** on 20,305 test records, with approximately 50% recall for Serious/Fatal. CarApp **retrains and evaluates the same model on the uploaded CSV**, rather than relying on a separate `.joblib` model file. Results may differ if different data are uploaded.

**Interpretation:** these are classifications of severity *among recorded injury collisions*. They are not estimates of the probability of an accident or reliable individual risk predictions. The model has substantial errors and is intended for coursework exploration.

## Project submission

Dr Mahmood's updated instructions require a ZIP submission containing the **Word report**, **Jupyter notebook**, **dataset**, and **Streamlit application folder**. This download contains only the **Streamlit application component**. Add your notebook, report and dataset separately to the final university submission ZIP.
