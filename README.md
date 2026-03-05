# DL-Project Setup on PACE ICE

This guide explains how to set up the environment and download the dataset for the project on the PACE ICE cluster.

---

## 1. Clone the repository

Log into PACE ICE and navigate to your scratch directory:

```bash
cd /scratch
git clone https://github.com/adibiasio/dl-project.git
cd dl-project
```

## 2. Configure Kaggle API

1. Go to [Kaggle](https://www.kaggle.com/) and log in.
2. Navigate to **Account** → **API** → **Create New API Token**.
3. Open `setup.sh` in your editor and replace the placeholder `KAGGLE_API_TOKEN` with your token from `kaggle.json`.

## 3. Run the setup script

From the project directory in your scratch folder, run:

```bash
bash setup.sh
```

1. This script will:

2. Install uv (a Python environment manager).

3. Create a virtual environment (.venv) in your project folder.

4. Install all Python dependencies.

5. Download the Facebook Hateful Meme dataset from Kaggle.