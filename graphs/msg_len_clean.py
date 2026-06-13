import pandas as pd
import matplotlib.pyplot as plt

# -------- SETTINGS --------
DATA_PATH = "../processed_data/sms_spam_processed.csv"

# -------- LOAD DATA --------
print("\nLoading data...")
df = pd.read_csv(DATA_PATH)

# Fix missing values for both columns
df["message_clean_default"] = df["message_clean_default"].fillna("")
df["message_clean_custom"] = df["message_clean_custom"].fillna("")

# -------- CLEAN LENGTHS --------
print("Computing clean message lengths...")
df["len_default"] = df["message_clean_default"].apply(len)
df["len_custom"] = df["message_clean_custom"].apply(len)

# -------- PLOT CONFIGURE --------
print("Plotting distributions...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

variants = [
    {"name": "Default Pipeline", "col": "len_default", "ax": ax1},
    {"name": "Custom Pipeline (with tokens)", "col": "len_custom", "ax": ax2}
]

# Find global max length to keep X-axis scales consistent
global_max = max(df["len_default"].max(), df["len_custom"].max())

for var in variants:
    # Split by label
    ham = df[df["target_encoded"] == 0][var["col"]]
    spam = df[df["target_encoded"] == 1][var["col"]]

    ax = var["ax"]
    ax.hist(ham, bins=50, range=(0, global_max), alpha=0.6, label="Ham (0)", color="teal")
    ax.hist(spam, bins=50, range=(0, global_max), alpha=0.6, label="Spam (1)", color="crimson")

    ax.set_title(f"Clean Length: {var['name']}")
    ax.set_xlabel("Message Length")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("\nDONE")