from load_labmt import load_labmt
from quantitative_exploration import analyse_disagreement 
from quantitative_exploration import save_csv
from quantitative_exploration import save_figure
import pandas as pd
import matplotlib.pyplot as plt



df = load_labmt()

# keep only rows that have a happiness score
df = df.dropna(subset=["happiness_rank"])

# 5 very positive (highest happiness)
very_positive = (
    df.sort_values("happiness_average", ascending=False)
      .head(10)
      .assign(category="very positive")
)

# 5 very negative (lowest happiness)
very_negative = (
    df.sort_values("happiness_average", ascending=True)
      .head(10)
      .assign(category="very negative")
)

#5 highly contested (high standard deviation)
highly_contested = analyse_disagreement(df.sort_values("happiness_standard_deviation", ascending=False)
        .head(10)
    )

polar = (
   df[df["happiness_average"].between(4.5, 5.5)]
   .sort_values("happiness_standard_deviation", ascending=False)
   .head(10)
   .assign(category="polarizing")
)

print("Very positive:\n", very_positive[["word", "happiness_average"]])
print("\nVery negative:\n", very_negative[["word", "happiness_average"]])
print("\nHighly contested:\n", highly_contested[["word", "happiness_average", "happiness_standard_deviation"]])
print("\nPolarizing:\n", polar[["word", "happiness_average", "happiness_standard_deviation"]])

# reset index so they can be aligned row-wise
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

save_csv(word_exhibit, "labmt_word_exhibit", index=False)

vp_tbl = very_positive[["word", "happiness_average"]]
vn_tbl = very_negative[["word", "happiness_average"]]
hc_tbl = highly_contested[["word", "happiness_average", "happiness_standard_deviation"]]
p_tbl  = polar[["word", "happiness_average", "happiness_standard_deviation"]]

tables = [
    ("Very positive", vp_tbl),
    ("Very negative", vn_tbl),
    ("Highly contested", hc_tbl),
    ("Polarizing", p_tbl),
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

save_figure("labmt_top_10_per_cat")

fig, ax = plt.subplots(figsize=(10, 3))
ax.axis("off")
ax.set_title("labmt_word_exhibit", fontweight="bold", pad=10, y=1.05)

table = ax.table(
    cellText=word_exhibit.values,
    colLabels=word_exhibit.columns,
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

plt.tight_layout()

save_figure("labmt_word_exhibit")