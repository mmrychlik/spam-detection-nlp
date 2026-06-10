import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# -------- SETTINGS --------
DATA_PATH = "./processed_data/sms_spam_processed.csv"

# -------- LOAD DATA --------
print("\nLoading data...")

df = pd.read_csv(DATA_PATH)
df["message_clean"] = df["message_clean"].fillna("")

X_pure = df["message_clean"].values
y = df["target_encoded"].values

# -------- VECTORIZERS --------
# -------- EXPERIMENT 1: CountVectorizer (Standard) --------
print("\nRunning CountVectorizer...")
count_vec = CountVectorizer(min_df=2)
X_count = count_vec.fit_transform(X_pure).toarray()
print(f"CountVectorizer Shape: {X_count.shape}")

# -------- EXPERIMENT 2: TfidfVectorizer (Standard) --------
print("\nRunning TfidfVectorizer...")
tfidf_vec = TfidfVectorizer(min_df=2)
X_tfidf = tfidf_vec.fit_transform(X_pure).toarray()
print(f"TF-IDF Shape: {X_tfidf.shape}")

# -------- EXPERIMENT 3: CountVectorizer (with Bigrams) --------
print("\nRunning CountVectorizer with Bigrams...")
count_ngram = CountVectorizer(min_df=2, ngram_range=(1, 2))
X_count_ngram = count_ngram.fit_transform(X_pure).toarray()
print(f"CountVectorizer (with Bigrams) Shape: {X_count_ngram.shape}")

# -------- EXPERIMENT 4: TfidfVectorizer (with Bigrams) --------
print("\nRunning TfidfVectorizer with Bigrams...")
tfidf_ngram = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
X_tfidf_ngram = tfidf_ngram.fit_transform(X_pure).toarray()
print(f"TF-IDF (with Bigrams) Shape: {X_tfidf_ngram.shape}")