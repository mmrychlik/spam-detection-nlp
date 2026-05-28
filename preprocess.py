import pandas as pd
import zipfile
import os
import re
import string
import html
import nltk

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
    import shutil
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

# -------- LINK HANDLING MENU --------
print("\nLink handling:")
print("1 - Remove links")
print("2 - Convert links to 'http' token")

choice = input("Choose option (1/2): ").strip()

PROCESS_LINKS = choice == "2"

# -------- TEXT CLEANING --------
print("Cleaning text...")


def clean_text(text):
    # 1. Cast to string FIRST to avoid errors if a row is NaN/null
    text = str(text)

    # 2. Unescape HTML entities (Safely translates &lt;#&gt; to <#>)
    text = html.unescape(text)

    # 3. Fix mojibake apostrophes BEFORE lowercasing
    text = text.replace('‰Û÷', "'").replace('‰Ûª', "'")

    # 4. Remove dataset-specific privacy artifacts
    text = text.replace('<#>', ' ').replace('ltgt', ' ')

    # 5. Lowercase
    text = text.lower()

    # 6. Remove text in brackets and stray HTML tags
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'<.*?>+', ' ', text)

    # 7. Handle links based on global PROCESS_LINKS flag
    link_pattern = r'https?://\S+|www\.\S+|\S+\.com\S*'
    if PROCESS_LINKS:
        text = re.sub(link_pattern, ' http ', text)
    else:
        text = re.sub(link_pattern, ' ', text)

    # -------------------------------------------------------------
    # 8. THE CRITICAL FIX: Handle Currency BEFORE Punctuation
    # -------------------------------------------------------------
    text = re.sub(r'(å£|£|\$|€|¥)', ' $$$ ', text)

    # 9. Strip punctuation EXCEPT the dollar sign (so $$$ survives)
    custom_punctuation = string.punctuation.replace('$', '')
    text = re.sub(r'[%s]' % re.escape(custom_punctuation), ' ', text)

    # 10. Remove line breaks and alphanumeric numbers (e.g., "win100")
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', ' ', text)

    # 11. Final Cleanup: Collapse multiple spaces into a single space
    # (Because replacing everything with ' ' creates huge whitespace gaps)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

df["message_clean"] = df["message"].apply(clean_text)

# -------- STOPWORDS --------
print("Removing stopwords...")

stop_words = stopwords.words("english")
more_stopwords = ["u", "im", "c"]
stop_words = stop_words + more_stopwords

def remove_stopwords(text):
    return " ".join(
        word for word in text.split()
        if word not in stop_words
    )

df["message_clean"] = df["message_clean"].apply(remove_stopwords)

# -------- STEMMING --------
print("Stemming text...")

stemmer = SnowballStemmer("english")

def stem_text(text):
    return " ".join(stemmer.stem(word) for word in text.split())

df["message_clean"] = df["message_clean"].apply(stem_text)

# -------- LABEL ENCODING --------
print("Encoding labels...")

le = LabelEncoder()
df["target_encoded"] = le.fit_transform(df["target"])

# -------- FINAL CLEANUP --------
df = df.drop_duplicates().reset_index(drop=True)

# -------- SAVE --------
df.to_csv(OUTPUT_SMS, index=False)

print("\nDONE")
print(f"Total messages: {len(df)}")
print(f"Ham/Spam distribution:\n{df['target_encoded'].value_counts()}")
print(f"Saved to: {OUTPUT_SMS}")

import shutil
shutil.rmtree(TEMP_DIR)