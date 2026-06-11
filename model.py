import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# -------- SETTINGS --------
DATA_PATH = "./processed_data/sms_spam_processed.csv"

# -------- LOAD DATA --------
print("\nLoading data...")
df = pd.read_csv(DATA_PATH)
df["message_clean"] = df["message_clean"].fillna("")

X_pure = df["message_clean"].values
y = df["target_encoded"].values

# -------- VECTORIZERS --------
print("\nVectorizing datasets...")

count_vec = CountVectorizer(min_df=2)
X_count = count_vec.fit_transform(X_pure)

tfidf_vec = TfidfVectorizer(min_df=2)
X_tfidf = tfidf_vec.fit_transform(X_pure)

count_ngram = CountVectorizer(min_df=2, ngram_range=(1, 2))
X_count_ngram = count_ngram.fit_transform(X_pure)

tfidf_ngram = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
X_tfidf_ngram = tfidf_ngram.fit_transform(X_pure)

datasets = {
    "CountVectorizer": X_count,
    "TfidfVectorizer": X_tfidf,
    "CountVectorizer (Bigrams)": X_count_ngram,
    "TfidfVectorizer (Bigrams)": X_tfidf_ngram
}

# -------- MODELS --------
models = {
    "MultinomialNB": MultinomialNB(),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "LinearSVC": LinearSVC(random_state=42)
}

# -------- CROSS-VALIDATION SETUP --------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']

# -------- EXPERIMENT RUNNER --------
print("\nStarting Cross-Validation Experiments...")
print("-" * 80)

results_list = []

for dataset_name, X_data in datasets.items():
    print(f"\nEvaluating dataset: {dataset_name} (Shape: {X_data.shape})")

    for model_name, model in models.items():
        scores = cross_validate(
            model,
            X_data,
            y,
            cv=cv,
            scoring=scoring_metrics,
            n_jobs=-1
        )

        mean_accuracy = np.mean(scores['test_accuracy'])
        mean_precision = np.mean(scores['test_precision'])
        mean_recall = np.mean(scores['test_recall'])
        mean_f1 = np.mean(scores['test_f1'])

        print(
            f"  > {model_name:<20} | F1: {mean_f1:.4f} | Precision: {mean_precision:.4f} | Recall: {mean_recall:.4f} | Acc: {mean_accuracy:.4f}")

        results_list.append({
            "Vectorizer": dataset_name,
            "Model": model_name,
            "F1-Score": mean_f1,
            "Precision": mean_precision,
            "Recall": mean_recall,
            "Accuracy": mean_accuracy
        })

# -------- SUMMARY --------
print("\n" + "=" * 35 + " FINAL LEADERBOARD (Ranked by F1) " + "=" * 35)
results_df = pd.DataFrame(results_list)
results_df = results_df.sort_values(by="F1-Score", ascending=False).reset_index(drop=True)
print(results_df.to_string(index=False))