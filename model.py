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
# we have 2902 features with at least 2 occurrences
# -------- EXPERIMENT 1: CountVectorizer (Bag of Words) --------
print("\nRunning CountVectorizer...")
count_vec = CountVectorizer(max_features=2902)
X_count = count_vec.fit_transform(X_pure).toarray()
print(f"CountVectorizer Shape: {X_count.shape}")

# -------- EXPERIMENT 2: TfidfVectorizer (Standard) --------
print("\nRunning TfidfVectorizer...")
tfidf_vec = TfidfVectorizer(max_features=2902)
X_tfidf = tfidf_vec.fit_transform(X_pure).toarray()
print(f"TF-IDF Shape: {X_tfidf.shape}")

# -------- EXPERIMENT 3: TfidfVectorizer (with Bigrams) --------
print("\nRunning TF-IDF with Unigrams + Bigrams...")
tfidf_ngram = TfidfVectorizer(max_features=2902, ngram_range=(1, 2))
X_ngram = tfidf_ngram.fit_transform(X_pure).toarray()
print(f"TF-IDF (N-gram) Shape: {X_ngram.shape}")