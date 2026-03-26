# src/analysis.py

"""
Statistical analysis of Guardian happiness scores.

-   We treat the articles we collected as a sample from a hypothetical
    superpopulation of all possible articles that could have been written under
    the same conditions. The uncertainty in our estimates reflects the natural
    variation that would occur if we could repeatedly sample articles from this
    process.


Sampling plan:

- Sample: All Guardian articles retrieved from the API for the periods 2010‑13 and 2020‑23,
  filtered to sections "World news", "Politics", "Opinion", and with valid happiness scores 
  (see operalisation in src/compute_lambt_scores.py).

- Population: Conceptualised as a superpopulation of all possible articles that could have
  been written under similar editorial policies. We use bootstrapping to quantify uncertainty
  due to sampling variability.

  
Methodology:

- Descriptive statistics (mean, median, SD, IQR, skewness, kurtosis) for each section‑period group.

- Bootstrap 95% confidence intervals for differences in means between:
    a) periods (overall)
    b) pairs of sections (ignoring period)
    c) periods within each section

- Inference plots: density by period, density by section, boxplot(change: density?) by section+period,
  and point‑with‑error‑bar plot of means with CIs.

- Coverage check: average proportion of matched words per group.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def compute_descriptive_stats(df):
    """Return a DataFrame with descriptive stats (mean, median, SD, IQR, skew, kurtosis) by group."""
    desc = df.groupby(["section_name", "period"])["happiness"].describe(percentiles=[.25, .5, .75])
    desc["skew"] = df.groupby(["section_name", "period"])["happiness"].apply(lambda x: x.skew())
    desc["kurtosis"] = df.groupby(["section_name", "period"])["happiness"].apply(lambda x: x.kurtosis())
    return desc


def bootstrap_diff(data1, data2, n_boot=10000, return_all=False):
    """
    Bootstrap difference of means (data2 - data1).
    If return_all=False: returns mean difference and 95% CI.
    If return_all=True: returns mean difference, 95% CI, and the full array of bootstrap diffs.
    """
    diffs = []
    for _ in range(n_boot):
        boot1 = np.random.choice(data1, size=len(data1), replace=True)
        boot2 = np.random.choice(data2, size=len(data2), replace=True)
        diffs.append(boot2.mean() - boot1.mean())
    ci_low = np.percentile(diffs, 2.5)
    ci_high = np.percentile(diffs, 97.5)
    if return_all:
        return np.mean(diffs), ci_low, ci_high, np.array(diffs)
    else:
        return np.mean(diffs), ci_low, ci_high


def plot_mean_ci(df, save_path):
    """Generate point+error bar plot of mean happiness by section and period with 95% CI."""
    grouped = df.groupby(["section_name", "period"])["happiness"]
    ci_data = []
    for (section, period), group in grouped:
        if len(group) < 5:
            continue
        boot_means = [np.mean(np.random.choice(group, size=len(group), replace=True)) for _ in range(5000)]
        ci_low = np.percentile(boot_means, 2.5)
        ci_high = np.percentile(boot_means, 97.5)
        ci_data.append((section, period, group.mean(), ci_low, ci_high))

    ci_df = pd.DataFrame(ci_data, columns=["section", "period", "mean", "ci_low", "ci_high"])
    # Plot
    plt.figure(figsize=(8,5))
    for period in ci_df["period"].unique():
        subset = ci_df[ci_df["period"] == period]
        plt.errorbar(subset["mean"], subset["section"],
                     xerr=[subset["mean"]-subset["ci_low"], subset["ci_high"]-subset["mean"]],
                     fmt='o', label=period, capsize=4)
    plt.xlabel("Mean happiness (with 95% CI)")
    plt.ylabel("Section")
    plt.title("Mean happiness by section and period with 95% CI")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved point+error plot to {save_path}")
    plt.show()

def plot_bootstrap_means_density(df, group_var, period_var, value_col, save_path, n_boot=10000):
    """
    Plot density of bootstrapped means for each group defined by group_var and period_var.
    If period_var is None, groups are only by group_var (ignoring period).
    """
    groups = df[group_var].unique()
    periods = df[period_var].unique() if period_var else [None]
    
    plt.figure(figsize=(8,5))
    
    for period in periods:
        for group in groups:
            if period:
                data = df[(df[group_var] == group) & (df[period_var] == period)][value_col].values
                label = f"{group} ({period})"
            else:
                data = df[df[group_var] == group][value_col].values
                label = group
            if len(data) < 5:
                print(f"Not enough data for {label}, skipping.")
                continue
            
            # Bootstrap means
            boot_means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
            
            # Plot density
            plt.hist(boot_means, bins=40, density=True, alpha=0.3, label=label)  # density=True for probability density
    
    plt.xlabel(f"Bootstrapped mean happiness")
    plt.ylabel("Density")
    plt.title(f"Bootstrap distributions of means\nby {group_var}" + (f" and {period_var}" if period_var else ""))
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved bootstrap means density plot to {save_path}")
    plt.show()


def plot_bootstrap_distribution(boot_diffs, observed_diff, ci_low, ci_high, save_path):
    """Plot histogram of bootstrap differences with CI and observed difference."""
    plt.figure(figsize=(8,5))
    plt.hist(boot_diffs, bins=40, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(observed_diff, color='red', linestyle='-', linewidth=2, 
                label=f'Observed diff = {observed_diff:.3f}')
    plt.axvline(ci_low, color='darkblue', linestyle='--', linewidth=2, 
                label=f'2.5% CI = {ci_low:.3f}')
    plt.axvline(ci_high, color='darkblue', linestyle='--', linewidth=2, 
                label=f'97.5% CI = {ci_high:.3f}')
    plt.xlabel('Bootstrap difference (2020‑23 minus 2010‑13)')
    plt.ylabel('Frequency')
    plt.title('Bootstrap distribution of the overall period difference')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved bootstrap distribution plot to {save_path}")
    plt.show()

# ----------------------------------------------------------------------
# Combined plots for comparisons
# ----------------------------------------------------------------------

def plot_combined_comparison1(boot_diffs, diff_mean, ci_low, ci_high, data_2010, data_2020, save_path):
    """Combined plot: histogram of differences + density of period means."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: bootstrap distribution of difference
    axes[0].hist(boot_diffs, bins=40, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].axvline(diff_mean, color='red', linestyle='-', linewidth=2, label=f'Observed diff = {diff_mean:.3f}')
    axes[0].axvline(ci_low, color='darkblue', linestyle='--', linewidth=2, label=f'2.5% CI = {ci_low:.3f}')
    axes[0].axvline(ci_high, color='darkblue', linestyle='--', linewidth=2, label=f'97.5% CI = {ci_high:.3f}')
    axes[0].set_xlabel('Bootstrap difference (2020‑23 minus 2010‑13)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Bootstrap distribution of overall period difference')
    axes[0].legend()
    
    # Right: bootstrap densities of means for the two periods
    boot_means_2010 = [np.mean(np.random.choice(data_2010, size=len(data_2010), replace=True)) for _ in range(10000)]
    boot_means_2020 = [np.mean(np.random.choice(data_2020, size=len(data_2020), replace=True)) for _ in range(10000)]
    axes[1].hist(boot_means_2010, bins=40, density=True, alpha=0.4, label='2010-13', color='blue')
    axes[1].hist(boot_means_2020, bins=40, density=True, alpha=0.4, label='2020-23', color='orange')
    axes[1].set_xlabel('Bootstrapped mean happiness')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Bootstrap distributions of period means')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved combined comparison 1 plot to {save_path}")

def plot_combined_comparison2(data_by_section, sections, save_path):
    """3‑panel figure with bootstrap distributions of differences for each pair."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    pairs = [(0,1), (0,2), (1,2)]
    for i, (i1, i2) in enumerate(pairs):
        sec1, sec2 = sections[i1], sections[i2]
        # Bootstrap differences (sec2 - sec1)
        diffs = []
        for _ in range(10000):
            boot1 = np.random.choice(data_by_section[sec1], size=len(data_by_section[sec1]), replace=True)
            boot2 = np.random.choice(data_by_section[sec2], size=len(data_by_section[sec2]), replace=True)
            diffs.append(boot2.mean() - boot1.mean())
        diff_mean = np.mean(diffs)
        ci_low = np.percentile(diffs, 2.5)
        ci_high = np.percentile(diffs, 97.5)
        axes[i].hist(diffs, bins=40, alpha=0.7, color='skyblue', edgecolor='black')
        axes[i].axvline(diff_mean, color='red', linestyle='-', linewidth=2, label=f'Mean diff = {diff_mean:.3f}')
        axes[i].axvline(ci_low, color='darkblue', linestyle='--', linewidth=2, label=f'2.5% CI = {ci_low:.3f}')
        axes[i].axvline(ci_high, color='darkblue', linestyle='--', linewidth=2, label=f'97.5% CI = {ci_high:.3f}')
        axes[i].set_xlabel(f'{sec2} minus {sec1}')
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'{sec1} vs {sec2}')
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved combined comparison 2 plot to {save_path}")

def plot_combined_comparison3(df, save_path):
    """Combine bootstrap densities (section×period) and point+CI plot."""
    from scipy.stats import gaussian_kde  # import here for clarity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: bootstrap densities
    groups = df.groupby(["section_name", "period"])["happiness"]
    # Colour mapping for section-period groups
    colors = {
        ('World news', '2010-2013'): 'blue',
        ('World news', '2020-2023'): 'navy',
        ('Politics', '2010-2013'): 'green',
        ('Politics', '2020-2023'): 'darkgreen',
        ('Opinion', '2010-2013'): 'orange',
        ('Opinion', '2020-2023'): 'darkorange'
    }
    for (section, period), group in groups:
        if len(group) < 5:
            continue
        boot_means = [np.mean(np.random.choice(group, size=len(group), replace=True)) for _ in range(10000)]
        ax1.hist(boot_means, bins=40, density=True, alpha=0.3,
                 color=colors.get((section, period), 'gray'),
                 label=f'{section} {period}')
    ax1.set_xlabel('Bootstrapped mean happiness')
    ax1.set_ylabel('Density')
    ax1.set_title('Bootstrap distributions of means')
    ax1.legend()
    
    # Right: point+CI plot
    ci_data = []
    for (section, period), group in groups:
        if len(group) < 5:
            continue
        boot_means = [np.mean(np.random.choice(group, size=len(group), replace=True)) for _ in range(5000)]
        ci_low = np.percentile(boot_means, 2.5)
        ci_high = np.percentile(boot_means, 97.5)
        ci_data.append((section, period, group.mean(), ci_low, ci_high))
    ci_df = pd.DataFrame(ci_data, columns=["section", "period", "mean", "ci_low", "ci_high"])
    for period in ci_df["period"].unique():
        subset = ci_df[ci_df["period"] == period]
        ax2.errorbar(subset["mean"], subset["section"],
                     xerr=[subset["mean"]-subset["ci_low"], subset["ci_high"]-subset["mean"]],
                     fmt='o', label=period, capsize=4)
    ax2.set_xlabel("Mean happiness (with 95% CI)")
    ax2.set_ylabel("Section")
    ax2.set_title("Mean happiness by section and period")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved combined comparison 3 plot to {save_path}")

def plot_ridgeline(df, save_path):
    """Create a ridgeline (joy) plot for happiness distributions by section and period."""
    from scipy.stats import gaussian_kde
    groups = df.groupby(["section_name", "period"])["happiness"]
    group_list = []
    for (section, period), data in groups:
        if len(data) >= 5:
            group_list.append((section, period, data))
    # Order groups (you can customise order, here we keep as is)
    group_list.sort(key=lambda x: (x[0], x[1]))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    y_offsets = np.arange(len(group_list))  # vertical positions
    for i, (section, period, data) in enumerate(group_list):
        # Compute KDE
        kde = gaussian_kde(data)
        x_vals = np.linspace(data.min(), data.max(), 200)
        y_vals = kde(x_vals)
        # Scale density to fit in a band (max height 0.8)
        y_vals = y_vals / y_vals.max() * 0.8
        # Plot filled area
        ax.fill_between(x_vals, i, i + y_vals, alpha=0.6, label=f"{section} {period}")
        # Add mean and median lines
        mean_val = data.mean()
        median_val = data.median()
        # Draw horizontal line at baseline
        ax.axhline(i, color='gray', linewidth=0.5)
        # Add mean and median as vertical lines
        ax.axvline(mean_val, ymin=i/len(group_list), ymax=(i+0.8)/len(group_list),
                   color='red', linestyle='-', linewidth=1.5, alpha=0.7)
        ax.axvline(median_val, ymin=i/len(group_list), ymax=(i+0.8)/len(group_list),
                   color='darkblue', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_yticks(y_offsets + 0.4)
    ax.set_yticklabels([f"{s} {p}" for s, p, _ in group_list])
    ax.set_xlabel("Happiness score")
    ax.set_ylabel("")
    ax.set_title("Ridgeline plot: happiness distributions by section and period\n(red line = mean, blue dashed = median)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved ridgeline plot to {save_path}")

def main():
    # 1. Load and prepare data
    data_path = Path("data/processed/guardian_articles_with_scores.csv")
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["happiness"]).copy()
    target_sections = ["World news", "Politics", "Opinion"]
    df = df[df["section_name"].isin(target_sections)].copy()

    print("Data loaded. Number of articles per group:")
    print(df.groupby(["section_name", "period"]).size())
    print()

    # 2. Descriptive statistics
    desc = compute_descriptive_stats(df)
    print("Descriptive statistics (point estimates) by group:")
    print(desc.round(3))
    print("\n" + "="*70 + "\n")

    # 3. Period difference overall (with full bootstrap distribution)
    data_2010 = df[df["period"] == "2010-2013"]["happiness"].values
    data_2020 = df[df["period"] == "2020-2023"]["happiness"].values
    diff_overall, ci_lo_overall, ci_hi_overall, boot_diffs = bootstrap_diff(
        data_2010, data_2020, n_boot=10000, return_all=True
    )

    print("COMPARISON 1: Happiness difference between periods (2020‑23 minus 2010‑13)")
    print(f"Mean difference = {diff_overall:.3f}")
    print(f"95% Bootstrap CI = [{ci_lo_overall:.3f}, {ci_hi_overall:.3f}]")
    print()

    # Compute Cohen's d for overall period difference
    n1, n2 = len(data_2010), len(data_2020)
    var1, var2 = np.var(data_2010, ddof=1), np.var(data_2020, ddof=1)
    pooled_sd = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    cohens_d = diff_overall / pooled_sd
    print(f"Cohen's d = {cohens_d:.3f}")

    # Plot the bootstrap distribution
    plot_bootstrap_distribution(boot_diffs, diff_overall, ci_lo_overall, ci_hi_overall,
                                Path("figures/bootstrap_period_difference.png"))

    # 4. Pairwise section differences
    sections = ["World news", "Politics", "Opinion"]
    data_by_section = {sec: df[df["section_name"] == sec]["happiness"].values for sec in sections}
    print("COMPARISON 2: Pairwise differences between sections")
    for i, sec1 in enumerate(sections):
        for sec2 in sections[i+1:]:
            diff, ci_lo, ci_hi, boot_diffs = bootstrap_diff(data_by_section[sec1], data_by_section[sec2], return_all=True)
            prob_sec2_gt_sec1 = np.mean(boot_diffs > 0)
            print(f"{sec1} vs {sec2}: difference = {diff:.3f}  [95% CI {ci_lo:.3f}, {ci_hi:.3f}]  Pr({sec2} > {sec1}) = {prob_sec2_gt_sec1:.3f}")
    print()



    # 5. Period difference within each section
    print("COMPARISON 3: Period difference within each section")
    for sec in sections:
        data_2010_sec = df[(df["section_name"] == sec) & (df["period"] == "2010-2013")]["happiness"].values
        data_2020_sec = df[(df["section_name"] == sec) & (df["period"] == "2020-2023")]["happiness"].values
        if len(data_2010_sec) < 5 or len(data_2020_sec) < 5:
            print(f"{sec}: insufficient data (n<5)")
            continue
        diff, ci_lo, ci_hi = bootstrap_diff(data_2010_sec, data_2020_sec)
        print(f"{sec}: change = {diff:.3f}  [95% CI {ci_lo:.3f}, {ci_hi:.3f}]")
    print()

    # 6. Point+error plot
    plot_mean_ci(df, Path("figures/mean_ci_by_section_period.png"))

    # 7. Coverage check
    print("\nAverage coverage by section and period:")
    coverage = df.groupby(["section_name", "period"])["labmt_proportion"].mean().unstack()
    print(coverage.round(3))
    print()

    # 8. Save all results to a text file
    output_file = Path("tables/analysis_results.txt")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("ANALYSIS RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write("Sampling plan:\n")
        f.write("Sample: Guardian articles from periods 2010‑13 and 2020‑23, sections 'World news', 'Politics', 'Opinion', with valid happiness scores.\n")
        f.write("Population: Hypothetical superpopulation of all possible articles under similar conditions.\n\n")
        f.write("Descriptive statistics by group:\n")
        f.write(desc.round(3).to_string())
        f.write("\n\n")
        f.write("Comparison 1: Period difference overall\n")
        f.write(f"Mean diff = {diff_overall:.3f}, 95% CI [{ci_lo_overall:.3f}, {ci_hi_overall:.3f}]\n\n")
        f.write("Comparison 2: Section differences\n")
        for i, sec1 in enumerate(sections):
            for sec2 in sections[i+1:]:
                diff, ci_lo, ci_hi = bootstrap_diff(data_by_section[sec1], data_by_section[sec2])
                f.write(f"{sec1} vs {sec2}: diff = {diff:.3f} [95% CI {ci_lo:.3f}, {ci_hi:.3f}]\n")
        f.write("\nComparison 3: Period difference within each section\n")
        for sec in sections:
            data_2010_sec = df[(df["section_name"] == sec) & (df["period"] == "2010-2013")]["happiness"].values
            data_2020_sec = df[(df["section_name"] == sec) & (df["period"] == "2020-2023")]["happiness"].values
            if len(data_2010_sec) >= 5 and len(data_2020_sec) >= 5:
                diff, ci_lo, ci_hi = bootstrap_diff(data_2010_sec, data_2020_sec)
                f.write(f"{sec}: change = {diff:.3f} [95% CI {ci_lo:.3f}, {ci_hi:.3f}]\n")
        f.write("\nCoverage:\n")
        f.write(coverage.round(3).to_string())

    print(f"\nAll results saved to {output_file}")

    # 9. Inferential density: bootstrap means by period (overall, ignoring section)
    plot_bootstrap_means_density(df, group_var="section_name", period_var="period", 
                                 value_col="happiness", 
                                 save_path=Path("figures/bootstrap_means_by_section_and_period.png"))

    # 10. Inferential density: bootstrap means by section (ignoring period)
    plot_bootstrap_means_density(df, group_var="section_name", period_var=None, 
                                 value_col="happiness", 
                                 save_path=Path("figures/bootstrap_means_by_section.png"))
    
     # 11. NEW: Inferential density: bootstrap means by period (pooled across sections)
    df_period_pooled = df.copy()
    df_period_pooled["period_pooled"] = df_period_pooled["period"]  # use period as grouping variable
    plot_bootstrap_means_density(df_period_pooled, group_var="period_pooled", period_var=None,
                                 value_col="happiness",
                                 save_path=Path("figures/bootstrap_means_by_period_pooled.png"))
    # Combined plots
    plot_combined_comparison1(boot_diffs, diff_overall, ci_lo_overall, ci_hi_overall,
                            data_2010, data_2020,
                            Path("figures/combined_comparison1.png"))
    plot_combined_comparison2(data_by_section, sections,
                            Path("figures/combined_comparison2.png"))
    plot_combined_comparison3(df, Path("figures/combined_comparison3.png"))


if __name__ == "__main__":
    main()