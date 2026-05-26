import pandas as pd
import matplotlib.pyplot as plt

# -------- SETTINGS --------
DATA_PATH = "../processed_data/sms_spam_processed.csv"

# -------- LOAD DATA --------
print("\nLoading data...")
df = pd.read_csv(DATA_PATH)

# -------- PLOT --------
print("Plotting label distribution...")
plt.figure(figsize=(6, 5))

label_counts = df["target_encoded"].value_counts()

labels = ["Ham (0)", "Spam (1)"]
counts = [label_counts.get(0, 0), label_counts.get(1, 0)]

plt.bar(labels, counts, color=["C0", "C1"], alpha=0.6, width=0.6)

plt.title("Label Distribution (Class Balance)")
plt.xlabel("Message Type")
plt.ylabel("Count")

for i, count in enumerate(counts):
    plt.text(i, count + (max(counts) * 0.01), str(count), ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("\nDONE")