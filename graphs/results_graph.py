import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------- SETTINGS --------
INPUT_RESULTS = "../processed_data/model_experiment_results.csv"

# -------- LOAD EXPORTED METRICS --------
try:
    results_df = pd.read_csv(INPUT_RESULTS)
    print(f"\nLoaded {len(results_df)} experiment configurations from history.")
except FileNotFoundError:
    raise FileNotFoundError(f"Missing '{INPUT_RESULTS}'. Please run model.py first.")

# -------- DYNAMIC COLUMNS PARSING --------
results_df["Ngram_Type"] = results_df["Vectorizer"].apply(
    lambda x: "Bigram" if "Bigrams" in str(x) else "Unigram"
)

results_df["Vectorizer_Short"] = results_df["Vectorizer"].replace({
    "CountVectorizer": "CountVec",
    "TfidfVectorizer": "TfidfVec",
    "CountVectorizer (Bigrams)": "CountVec (1,2)",
    "TfidfVectorizer (Bigrams)": "TfidfVec (1,2)"
})

pipeline_col = "Pipeline Setup"
model_col = "Model"
f1_col = "F1-Score"

# -------- STATIC DISPLAY RULES --------
STATIC_MODEL_ORDER = ["MultinomialNB", "LogisticRegression", "LinearSVC"]


STATIC_COLOR_PALETTE = {
    # Default (Reds)
    "CountVec": "#b04c4c",
    "TfidfVec": "#e09696",
    "CountVec (1,2)": "#b04c4c",
    "TfidfVec (1,2)": "#e09696",

    # Custom (Greens)
    "CountVec": "#4cb06a",
    "TfidfVec": "#96e0ab",
    "CountVec (1,2)": "#4cb06a",
    "TfidfVec (1,2)": "#96e0ab",
}

# -------- PLOTTING THE 4 COMPLEX GRAPHS --------
sns.set_theme(style="whitegrid")

plot_setups = [
    {
        "Pipeline": "Default",
        "Ngram_Type": "Unigram",
        "hue_order": ["CountVec", "TfidfVec"],
        "title": "Default Pipeline: Unigram Performance",
        "palette": {"CountVec": "#b04c4c", "TfidfVec": "#e09696"}
    },
    {
        "Pipeline": "Default",
        "Ngram_Type": "Bigram",
        "hue_order": ["CountVec (1,2)", "TfidfVec (1,2)"],
        "title": "Default Pipeline: Bigram Performance",
        "palette": {"CountVec (1,2)": "#b04c4c", "TfidfVec (1,2)": "#e09696"}
    },
    {
        "Pipeline": "Custom",
        "Ngram_Type": "Unigram",
        "hue_order": ["CountVec", "TfidfVec"],
        "title": "Custom Pipeline: Unigram Performance",
        "palette": {"CountVec": "#4cb06a", "TfidfVec": "#96e0ab"}
    },
    {
        "Pipeline": "Custom",
        "Ngram_Type": "Bigram",
        "hue_order": ["CountVec (1,2)", "TfidfVec (1,2)"],
        "title": "Custom Pipeline: Bigram Performance",
        "palette": {"CountVec (1,2)": "#4cb06a", "TfidfVec (1,2)": "#96e0ab"}
    }
]

for setup in plot_setups:
    filtered_df = results_df[
        (results_df[pipeline_col] == setup["Pipeline"]) &
        (results_df["Ngram_Type"] == setup["Ngram_Type"])
        ]

    if filtered_df.empty:
        continue

    plt.figure(figsize=(10, 6))

    ax = sns.barplot(
        data=filtered_df,
        x=model_col,
        y=f1_col,
        hue="Vectorizer_Short",
        order=STATIC_MODEL_ORDER,
        hue_order=setup["hue_order"],
        palette=setup["palette"],
        edgecolor="black"
    )

    # Cosmetics
    plt.title(setup["title"], fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Classification Models", fontsize=11, labelpad=8)
    plt.ylabel("Mean F1-Score (5-Fold CV)", fontsize=11, labelpad=8)

    # STRICT STATIC Y-AXIS LIMITS
    plt.ylim(0.75, 1.00)

    # Add precise value labels to the top of every bar
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', padding=4, fontsize=9, fontweight='semibold')

    # Legend locked to upper right corner
    plt.legend(title="Text Setup", loc="upper right", framealpha=0.9)
    plt.tight_layout()

print("\nRendering graphs...")
plt.show()
print("Done.")