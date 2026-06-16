import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

# -------- SETTINGS --------
DATA_PATH = "./processed_data/sms_spam_processed.csv"
OUTPUT_RESULTS = "./processed_data/model_experiment_results.csv"

# -------- LOAD DATA --------
print("\nLoading data...")
df = pd.read_csv(DATA_PATH)

df["message_clean_default"] = df["message_clean_default"].fillna("")
df["message_clean_custom"] = df["message_clean_custom"].fillna("")

y = df["target_encoded"].values

preprocessing_variants = {
    "Default": df["message_clean_default"].values,
    "Custom": df["message_clean_custom"].values
}

# -------- VECTORIZERS --------
vectorizers = {
    "CountVectorizer": CountVectorizer(min_df=5),
    "TfidfVectorizer": TfidfVectorizer(min_df=5),
    "CountVectorizer (Bigrams)": CountVectorizer(min_df=5, ngram_range=(1, 2)),
    "TfidfVectorizer (Bigrams)": TfidfVectorizer(min_df=5, ngram_range=(1, 2))
}

# -------- MODELS --------
models = {
    "MultinomialNB": MultinomialNB(),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "LinearSVC": LinearSVC(max_iter=2000, random_state=42)
}

# -------- CROSS-VALIDATION SETUP --------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring_metrics = ['balanced_accuracy', 'precision', 'recall', 'f1']

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

# -------- EXPERIMENT RUNNER --------
print("\nStarting Cross-Validation Experiments...")
print("-" * 95)

results_list = []

for prep_name, X_data in preprocessing_variants.items():
    print(f"\nEvaluating pipeline variant: [ {prep_name} ]")

    for vec_name, vec in vectorizers.items():
        for model_name, model in models.items():
            pipeline = Pipeline([
                ('vectorizer', vec),
                ('classifier', model)
            ])

            scores = cross_validate(
                pipeline,
                X_data,
                y,
                cv=cv,
                scoring=scoring_metrics,
                n_jobs=-1
            )

            y_pred = cross_val_predict(pipeline, X_data, y, cv=cv, n_jobs=-1)
            tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

            mean_accuracy = np.mean(scores['test_balanced_accuracy'])
            mean_precision = np.mean(scores['test_precision'])
            mean_recall = np.mean(scores['test_recall'])
            mean_f1 = np.mean(scores['test_f1'])

            print(f"  > {vec_name:<25} + {model_name:<20} | F1: {mean_f1:.4f}")

            results_list.append({
                "Pipeline Setup": prep_name,
                "Vectorizer": vec_name,
                "Model": model_name,
                "F1-Score": mean_f1,
                "Precision": mean_precision,
                "Recall": mean_recall,
                "Balanced Accuracy": mean_accuracy,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "TP": tp
            })

# -------- SUMMARY LEADERBOARD --------
print("\n" + "=" * 42 + " FINAL LEADERBOARD (Ranked by F1) " + "=" * 42)
results_df = pd.DataFrame(results_list)
results_df = results_df.sort_values(by="F1-Score", ascending=False).reset_index(drop=True)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(results_df.to_string(index=False))

# -------- SAVE RESULTS FOR GRAPHING --------
results_df.to_csv(OUTPUT_RESULTS, index=False)
print(f"\nSaved metrics database to: {OUTPUT_RESULTS}")