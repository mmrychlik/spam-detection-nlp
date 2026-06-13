import pandas as pd
import zipfile
import os
import re
import string
import html
import nltk
import shutil

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.preprocessing import LabelEncoder

# nltk.download("stopwords")

# -------- SETTINGS --------
RAW_DIR = "raw_data"
TEMP_DIR = "temp_extract"
OUTPUT_DIR = "processed_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_SMS = f"{OUTPUT_DIR}/sms_spam_processed.csv"

# -------- CLEAN FILE --------
if os.path.exists(OUTPUT_SMS):
    os.remove(OUTPUT_SMS)

if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)

os.makedirs(TEMP_DIR, exist_ok=True)

# -------- UNZIP DATASET --------
print("\nExtracting dataset...")

zip_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".zip")]

if not zip_files:
    raise FileNotFoundError("No ZIP file found in raw_data")

zip_path = os.path.join(RAW_DIR, zip_files[0])

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(TEMP_DIR)

print("Dataset extracted.")

# -------- LOAD DATA --------
print("\nLoading dataset...")

df = pd.read_csv(
    f"{TEMP_DIR}/spam.csv",
    encoding="latin-1"
)

df = df[["v1", "v2"]]
df.columns = ["target", "message"]
df["message_len"] = df["message"].apply(len)

# -------- TEXT CLEANING FUNCTION --------
def clean_text(text, custom_processing=False):
    text = str(text)
    text = html.unescape(text)

    text = text.replace('‰Û÷', "'").replace('‰Ûª', "'")
    text = text.replace('Û÷', "'")
    text = text.replace('åÕ', "'")
    text = text.replace('ÛÒ', " ")
    text = text.replace('Û', " ")
    text = text.replace('Ì', "I").replace('ì', "i")
    text = text.replace('<#>', ' ').replace('ltgt', ' ')

    text = text.lower()

    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'<.*?>+', ' ', text)

    link_pattern = r'https?://\S+|www\.\S+|\S+\.com\S*'

    # -------- CONDITIONAL FEATURE ENGINEERING --------
    if custom_processing:
        text = re.sub(link_pattern, ' http ', text)
        text = re.sub(r'(å£|£|\$|€|¥)', ' $$$ ', text)
    else:

        text = re.sub(link_pattern, ' ', text)

    custom_punctuation = string.punctuation.replace('$', '')
    text = re.sub(r'[%s]' % re.escape(custom_punctuation), ' ', text)

    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# -------- STOPWORDS & STEMMING HELPERS --------
stop_words = stopwords.words("english") + ["u", "im", "c"]
stemmer = SnowballStemmer("english")

def remove_stopwords(text):
    return " ".join(word for word in text.split() if word not in stop_words)

def stem_text(text):
    return " ".join(stemmer.stem(word) for word in text.split())

def pipeline_clean(series, custom_processing):
    """Applies the full cleaning, stopword removal, and stemming pipeline."""
    cleaned = series.apply(lambda x: clean_text(x, custom_processing=custom_processing))
    cleaned = cleaned.apply(remove_stopwords)
    return cleaned.apply(stem_text)

# -------- PROCESS BOTH VERSIONS --------
print("Processing text (Default pipeline)...")
df["message_clean_default"] = pipeline_clean(df["message"], custom_processing=False)

print("Processing text (Custom pipeline)...")
df["message_clean_custom"] = pipeline_clean(df["message"], custom_processing=True)

# -------- LABEL ENCODING & DUPLICATE DROPPING --------
print("Encoding labels...")
le = LabelEncoder()
df["target_encoded"] = le.fit_transform(df["target"])

df = df.drop_duplicates(subset=["message_clean_default", "message_clean_custom"]).reset_index(drop=True)

# -------- SAVE & CLEANUP --------
df.to_csv(OUTPUT_SMS, index=False)

print("\nDONE")
print(f"Total messages: {len(df)}")
print(f"Ham/Spam distribution:\n{df['target_encoded'].value_counts()}")
print(f"Saved to: {OUTPUT_SMS}")

shutil.rmtree(TEMP_DIR)