import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

# -------- SETTINGS --------
DATA_PATH = "./processed_data/sms_spam_processed.csv"

# -------- LOAD DATA --------
print("\nLoading data...")
df = pd.read_csv(DATA_PATH)
df["message_clean"] = df["message_clean"].fillna("")

X_pure = df["message_clean"].values
y = df["target_encoded"].values

# -------- VECTORIZERS --------
print("\nSetting up pipelines...")
vectorizers = {
    "CountVectorizer": CountVectorizer(min_df=2),
    "TfidfVectorizer": TfidfVectorizer(min_df=2),
    "CountVectorizer (Bigrams)": CountVectorizer(min_df=2, ngram_range=(1, 2)),
    "TfidfVectorizer (Bigrams)": TfidfVectorizer(min_df=2, ngram_range=(1, 2))
}

# -------- MODELS --------
models = {
    "MultinomialNB": MultinomialNB(),
    "LogisticRegression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    "LinearSVC": LinearSVC(class_weight='balanced', max_iter=2000, random_state=42)
}

# -------- CROSS-VALIDATION SETUP --------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']

# -------- EXPERIMENT RUNNER --------
print("\nStarting Cross-Validation Experiments...")
print("-" * 80)

results_list = []

for vec_name, vec in vectorizers.items():
    print(f"\nEvaluating vectorizer: {vec_name}")

    for model_name, model in models.items():
        pipeline = Pipeline([
            ('vectorizer', vec),
            ('classifier', model)
        ])

        scores = cross_validate(
            pipeline,
            X_pure,
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
            f"  > {model_name:<20} | F1: {mean_f1:.4f} | Precision: {mean_precision:.4f} | Recall: {mean_recall:.4f}")

        results_list.append({
            "Vectorizer": vec_name,
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