from load_labmt import load_labmt
from compute_labmt_scores import tokenize 
from quantitative_exploration import save_csv
from quantitative_exploration import save_figure
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

processed_path = Path("data/processed/guardian_articles_processed.csv")
df = pd.read_csv(processed_path)
print("Loaded processed articles:", df.shape)

labmt_path = Path("data/raw/Data_Set_S1.txt")
labmt_df = load_labmt(labmt_path)

word_col = "word"               
score_col = "happiness_average" 
std_col = "happiness_standard_deviation"

word_to_score = dict(zip(labmt_df[word_col], labmt_df[score_col]))
word_to_std = dict(zip(labmt_df[word_col], labmt_df[std_col]))
print(f"Loaded labMT lexicon with {len(word_to_score)} words")

    # 3. 对每篇文章计算 happiness + 命中词数
def compute_happiness(text: str):
        tokens = tokenize(text)
        scores = [word_to_score[w] for w in tokens if w in word_to_score]
        if not scores:
            return np.nan
        return float(np.mean(scores))

def count_labmt_words(text: str):
        tokens = tokenize(text)
        return sum(1 for w in tokens if w in word_to_score)

df["happiness"] = df["body_text"].apply(compute_happiness)
df["labmt_token_count"] = df["body_text"].apply(count_labmt_words)

print(df)

df["tokens"] = df["body_text"].apply(tokenize)
df["tokens_in_labmt"] = df["tokens"].apply(
    lambda toks: [w for w in toks if w in word_to_score]
)

word_df = df[["tokens_in_labmt"]].explode("tokens_in_labmt").rename(
    columns={"tokens_in_labmt": "word"}
)
word_df["happiness_score"] = word_df["word"].map(word_to_score)
word_df["happiness_std"] = word_df["word"].map(word_to_std)

print(word_df)
save_csv(word_df, "guardian_word_scores", index=False)

word_df = word_df.dropna(subset=["happiness_score"])

# 10 very positive (highest happiness)
very_positive = (
    word_df.sort_values("happiness_score", ascending=False).drop_duplicates()
      .head(10)
      .assign(category="very positive")
)

# 10 very negative (lowest happiness)
very_negative = (
    word_df.sort_values("happiness_score", ascending=True).drop_duplicates()
      .head(10)
      .assign(category="very negative")
)

# 10 highly contested (highest standard deviation)
highly_contested = (
    word_df.sort_values("happiness_std", ascending=False). drop_duplicates()
        .head(10)
        .assign(category="highly contested")
)

# 10 polarizing
polar = (
   word_df[word_df["happiness_score"].between(4.5, 5.5)].drop_duplicates()
   .sort_values("happiness_std", ascending=False)
   .head(10)
   .assign(category="polarizing")
)

print("Very positive:\n", very_positive[["word", "happiness_score"]])
print("\nVery negative:\n", very_negative[["word", "happiness_score"]])
print("\nHighly contested:\n", highly_contested[["word", "happiness_score", "happiness_std"]])
print("\nPolarizing:\n", polar[["word", "happiness_score", "happiness_std"]])


vp_words = very_positive["word"].reset_index(drop=True)
vn_words = very_negative["word"].reset_index(drop=True)
hc_words = highly_contested["word"].reset_index(drop=True)
p_words = polar["word"].reset_index(drop=True)

word_exhibit = pd.DataFrame({
    "very positive words": vp_words,
    "very negative words": vn_words,
    "highly contested words": hc_words,
    "polarizing words": p_words
})

print("Word exhibit:\n", word_exhibit)
print(word_exhibit.columns)

save_csv(word_exhibit, "guardian_word_exhibit", index=False)

vp_tbl = very_positive[["word", "happiness_score"]]
vn_tbl = very_negative[["word", "happiness_score"]]
hc_tbl = highly_contested[["word", "happiness_score", "happiness_std"]]
p_tbl  = polar[["word", "happiness_score", "happiness_std"]]

tables = [
    ("Very positive", vp_tbl),
    ("Very negative", vn_tbl),
     ("Highly contested", hc_tbl),
    ("Polarizing", p_tbl)
]

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

for ax, (title, df_sub) in zip(axes.flat, tables):
    ax.axis("off")            # no axes
    ax.set_title(title, fontweight="bold", pad=10)

    table = ax.table(
        cellText=df_sub.values,
        colLabels=df_sub.columns,
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)       # (width, height) scaling

plt.tight_layout()

save_figure("guardian_10_per_category")

fig, ax = plt.subplots(figsize=(10, 3))
ax.axis("off")
ax.set_title("Word exhibit", fontweight="bold", pad=10, y=1.05)

table = ax.table(
    cellText=word_exhibit.values,
    colLabels=word_exhibit.columns,
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

plt.tight_layout()

save_figure("guardian_word_exhibit")

