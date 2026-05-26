import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# -------- SETTINGS --------
DATA_PATH = "../processed_data/sms_spam_processed.csv"

# -------- LOAD DATA --------
print("\nLoading data...")

df = pd.read_csv(DATA_PATH)
df["message_clean"] = df["message_clean"].fillna("")

# -------- SPLIT DATA --------
print("Splitting data...")

ham_text = " ".join(df[df["target_encoded"] == 0]["message_clean"])
spam_text = " ".join(df[df["target_encoded"] == 1]["message_clean"])

# -------- WORDCLOUD GENERATION --------
print("Generating wordclouds...")

ham_wc = WordCloud(
    width=800,
    height=400,
    background_color="white",
    max_words=100
).generate(ham_text)

spam_wc = WordCloud(
    width=800,
    height=400,
    background_color="white",
    max_words=100
).generate(spam_text)

# -------- PLOT --------
print("Plotting...")

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.imshow(ham_wc, interpolation="bilinear")
plt.title("Ham WordCloud")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(spam_wc, interpolation="bilinear")
plt.title("Spam WordCloud")
plt.axis("off")

plt.tight_layout()
plt.show()

print("\nDONE")