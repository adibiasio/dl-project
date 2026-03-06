# DL-Project Setup on PACE ICE

This guide explains how to set up the environment and download the dataset for the project on the PACE ICE cluster.

---

## 1. Clone the repository

Log into PACE ICE and navigate to your scratch directory:

```bash
cd scratch
git clone https://github.com/adibiasio/dl-project.git
cd dl-project
```

## 2. Configure Kaggle API

1. Go to [Kaggle](https://www.kaggle.com/) and log in.
2. Navigate to **Account** → **API** → **Generate New Token**.
3. Create a `.env` file in your editor and copy the token into the file like this:
```bash
KAGGLE_API_TOKEN=your_kaggle_api_token
```

## 3. Configure Hugging Face API

1. Go to [Hugging Face](https://huggingface.co/) and log in.
2. Navigate to **Settings** → **Access Tokens**.
3. Click **Create new token**, make it a **Read** token, and give it a name (e.g., dl-project), and generate the token.
4. Add the token to your `.env` file:
```bash
HF_TOKEN=your_huggingface_token
```

## 4. Run the setup script

From the project directory in your scratch folder, run:

```bash
bash setup.sh
```

This script will:
1. Install uv (a Python environment manager).
2. Create a virtual environment (.venv) in your project folder.
3. Install all Python dependencies.
4. Download the Facebook Hateful Meme dataset from Kaggle.

## 5. Unzip the downloaded data

Unzip the downloaded `facebook-hateful-meme-dataset.zip`, which should now be visible in the project directory
```bash
unzip facebook-hateful-meme-dataset.zip
```