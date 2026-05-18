import pandas as pd
import matplotlib.pyplot as plt

# -------- SETTINGS --------
DATA_PATH = "processed_data/sms_spam_processed.csv"

# -------- LOAD DATA --------
print("\nLoading data...")

df = pd.read_csv(DATA_PATH)

# -------- RAW LENGTH --------
print("Computing raw message length...")

df["message_len_raw"] = df["message"].apply(len)

# split by label
ham = df[df["target_encoded"] == 0]["message_len_raw"]
spam = df[df["target_encoded"] == 1]["message_len_raw"]

# -------- PLOT --------
print("Plotting distribution...")

plt.figure(figsize=(10, 6))

max_len = max(ham.max(), spam.max())

plt.hist(ham, bins=50, range=(0, max_len), alpha=0.6, label="Ham (0)")
plt.hist(spam, bins=50, range=(0, max_len), alpha=0.6, label="Spam (1)")

plt.title("Raw Message Length: Ham vs Spam")
plt.xlabel("Message Length")
plt.ylabel("Frequency")

plt.legend()
plt.show()

print("\nDONE")