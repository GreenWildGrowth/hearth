# Climate analog cities

A small tool to explore **climate analogues between cities**.

👉 Given a city and a future climate scenario, the tool answers:

> “Which current cities have a climate similar to the future climate of this city?”

Inspired by the paper:  
**Understanding climate change from a global analysis of city analogues** (Crowther Lab)

---

## 🚀 Quick start

### 1. Install

```bash
git clone <repo>
cd <repo>

python3 -m venv virt_env
source virt_env/bin/activate

pip install -r requirements.txt
```

---

### 2. Run the pipeline

```bash
python3 run_pipeline.py --prefer-dedup
```

This will:

- download climate data  
- build a global city catalog  
- extract climate features  
- compute climate analogues  

---

### 3. Launch the app

```bash
streamlit run app_streamlit.py
```

---

## 🖥️ Using the app

- Select a **target city**
- View its **top climate analogues**
- Explore:
  - climate distance
  - geographic distance
  - urban similarity (population / capital)

---

## ⚠️ Known issue (Linux / Chrome)

On some Linux setups:

👉 the map may **not render in Chrome**

If that happens:

- use **Firefox** → works reliably  

This is a **browser/WebGL issue**, not a data issue.

---

## 📊 Outputs

Main result file:

```
data/processed/city_analogues_weighted_normalized.csv
```

Each row contains:

- a target city  
- its top analog cities  
- climate + geographic + urban distances  

---

## ⚙️ Configuration (advanced)

In `compute_city_analogues.py`:

```python
MATCH_MODE = "weighted_normalized"
URBAN_LAMBDA = 0.25
```

- Mostly climate-driven  
- Slight correction for city realism  

---

## 🧠 Method (short version)

- Climate = 19 BIOCLIM variables  
- PCA reduces dimensionality  
- Distance in PCA space = climate similarity  
- Optional urban penalty improves realism  

---

## 🔬 Work in progress

- Improve frontend (UX + map + storytelling)  
- Diversify analog selection (avoid clusters)  
- Introduce **north/south climate shift metric**  
- Multi-scenario support  

---

## 📚 Reference

> *Understanding climate change from a global analysis of city analogues*  
> Crowther Lab, ETH Zurich  
> https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0217592  

---

## 💡 TL;DR

```bash
pip install -r requirements.txt
python3 run_pipeline.py --prefer-dedup
streamlit run app_streamlit.py
```
