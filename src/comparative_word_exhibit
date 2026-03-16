import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from quantitative_exploration import save_csv
from quantitative_exploration import save_figure


ROOT = Path(__file__).resolve().parents[1]

FIGURES_DIR = ROOT / "figures"
TABLES_DIR = ROOT / "tables"

FIGURES_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)

# === 1. Set paths to your CSV files ===
# Example: your code is in /project/scripts/
# CSVs are in /project/data/

labmt_exhibit_path = f"{TABLES_DIR}/labmt_word_exhibit"  # adjust to your real path
guardian_exhibit_path = f"{TABLES_DIR}/guardian_word_exhibit"  # adjust to your real path

# === 2. Load the data ===
df_labmt = pd.read_csv(labmt_exhibit_path)
df_guardian = pd.read_csv(guardian_exhibit_path)

fig, ax = plt.subplots(figsize=(12, 7))  # adjust size as needed
ax.axis("off")  # hide axes

fig.suptitle("Comparative word exhibit: labMT 1.0 vs Guardian", fontsize=14, y=0.98)

labmt_bbox = [0, 0.52, 1, 0.40]
guardian_bbox = [0, 0.02, 1, 0.40]

table_labmt = ax.table(
    cellText=df_labmt.values,
    colLabels=df_labmt.columns,
    cellLoc="center",
    loc="center",
    bbox=labmt_bbox
)

table_guardian = ax.table(
    cellText=df_guardian.values,
    colLabels=df_guardian.columns,
    cellLoc="center",
    loc="center",
    bbox=guardian_bbox
)

ax.text(0.5, labmt_bbox[1] + labmt_bbox[3] + 0.03, "labMT 1.0",
        ha="center", va="bottom", fontsize=12, fontweight="bold")

ax.text(0.5, guardian_bbox[1] + guardian_bbox[3] + 0.03, "Guardian",
        ha="center", va="bottom", fontsize=12, fontweight="bold")

# Bold header row (row 0) in both tables
for col in range(len(df_labmt.columns)):
    table_labmt[(0, col)].set_text_props(fontweight="bold")

for col in range(len(df_guardian.columns)):
    table_guardian[(0, col)].set_text_props(fontweight="bold")

# Optional: adjust font size and scaling so 10 rows fit nicely
table_labmt.auto_set_font_size(False)
table_labmt.set_fontsize(8)
table_labmt.scale(1, 1)

table_guardian.auto_set_font_size(False)
table_guardian.set_fontsize(8)
table_guardian.scale(1, 1)

plt.tight_layout()

save_figure("comparative_word_exhibit")