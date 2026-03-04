from load_labmt import load_labmt
from quantitative_exploration import analyse_disagreement 
from quantitative_exploration import save_csv
import pandas as pd


df = load_labmt()

# keep only rows that have a happiness score
df = df.dropna(subset=["happiness_rank"])

# 5 very positive (highest happiness)
very_positive = (
    df.sort_values("happiness_rank", ascending=True)
      .head(5)
      .assign(category="very positive")
)

# 5 very negative (lowest happiness)
very_negative = (
    df.sort_values("happiness_rank", ascending=False)
      .head(5)
      .assign(category="very negative")
)

#5 highly contested (high standard deviation)
highly_contested = analyse_disagreement(df.sort_values("happiness_standard_deviation", ascending=False)
        .head(5)
    )

polar = (
   df[df["happiness_average"].between(4.5, 5.5)]
   .sort_values("happiness_rank", ascending=False)
   .head(5)
   .assign(category="polarizing")
)

print("Very positive:\n", very_positive[["word", "happiness_rank"]])
print("\nVery negative:\n", very_negative[["word", "happiness_rank"]])
print("\nHighly contested:\n", highly_contested[["word", "happiness_standard_deviation"]])
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

save_csv(word_exhibit, "Word exhibit", index=False)