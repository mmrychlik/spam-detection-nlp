import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

FILE_PATH = "../processed_data/model_experiment_results.csv"
df = pd.read_csv(FILE_PATH)

# select model index from result file
MODEL_INDEX = 0
# MODEL_INDEX = 8

selected_model = df.iloc[MODEL_INDEX]

pipeline_type = selected_model["Pipeline Setup"]
vectorizer_name = selected_model["Vectorizer"]
classifier_name = selected_model["Model"]
f1_score = float(selected_model["F1-Score"])
graph_title = f"{classifier_name} + {vectorizer_name}\n({pipeline_type} Pipeline) | F1-Score: {f1_score:.3f}"

TN = int(selected_model["TN"])
FP = int(selected_model["FP"])
FN = int(selected_model["FN"])
TP = int(selected_model["TP"])

print(f"Loaded index {MODEL_INDEX}: {classifier_name}")
print(f"TN: {TN}, FP: {FP}, FN: {FN}, TP: {TP}")

matrix = np.array([[TN, FP],
                   [FN, TP]])

sns.set_context("talk")
plt.figure(figsize=(8, 6))

sns.heatmap(matrix, annot=True, fmt="", cmap="Blues",
            cbar=False, linewidths=2, linecolor='black',
            xticklabels=["Ham", "Spam"],
            yticklabels=["Ham", "Spam"],
            annot_kws={"size": 20, "weight": "bold"})

plt.title(f"{graph_title}", fontsize=20, pad=20, weight="bold")
plt.xlabel("Predicted", fontsize=18, labelpad=15)
plt.ylabel("Actual", fontsize=18, labelpad=15)

plt.tight_layout()
plt.show()
