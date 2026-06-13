import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# -------- SETTINGS --------
DATA_PATH = "../processed_data/sms_spam_processed.csv"

# -------- LOAD DATA --------
print("\nLoading data...")
df = pd.read_csv(DATA_PATH)

df["message_clean_default"] = df["message_clean_default"].fillna("")
df["message_clean_custom"] = df["message_clean_custom"].fillna("")

# -------- WORDCLOUD GENERATION FUNCTION --------
def generate_wc(text_data):
    return WordCloud(
        width=800,
        height=400,
        background_color="white",
        max_words=100,
        stopwords={""},
        regexp=r"\w+|\$+"
    ).generate(text_data)

# -------- PREPARE TEXTS & GENERATE --------
print("Generating wordclouds for all configurations...")

# Default variant texts
ham_default = " ".join(df[df["target_encoded"] == 0]["message_clean_default"])
spam_default = " ".join(df[df["target_encoded"] == 1]["message_clean_default"])

# Custom variant texts
ham_custom = " ".join(df[df["target_encoded"] == 0]["message_clean_custom"])
spam_custom = " ".join(df[df["target_encoded"] == 1]["message_clean_custom"])

wc_ham_def = generate_wc(ham_default)
wc_spam_def = generate_wc(spam_default)
wc_ham_cust = generate_wc(ham_custom)
wc_spam_cust = generate_wc(spam_custom)

# -------- PLOT 2x2 GRID --------
print("Plotting...")
fig, axs = plt.subplots(2, 2, figsize=(16, 10))

# Row 1: Default
axs[0, 0].imshow(wc_ham_def, interpolation="bilinear")
axs[0, 0].set_title("Ham WordCloud (Default Pipeline)", fontsize=14)
axs[0, 1].imshow(wc_spam_def, interpolation="bilinear")
axs[0, 1].set_title("Spam WordCloud (Default Pipeline)", fontsize=14)

# Row 2: Custom
axs[1, 0].imshow(wc_ham_cust, interpolation="bilinear")
axs[1, 0].set_title("Ham WordCloud (Custom Pipeline)", fontsize=14)
axs[1, 1].imshow(wc_spam_cust, interpolation="bilinear")
axs[1, 1].set_title("Spam WordCloud (Custom Pipeline)", fontsize=14)

for ax in axs.flat:
    ax.axis("off")

plt.tight_layout()
plt.show()

print("\nDONE")