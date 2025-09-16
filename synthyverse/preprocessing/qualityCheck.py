import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew


# =====================
# 1. BASIC STATISTICS
# =====================
def calculate_basic_stats(df):
    empty_string_count = {}
    whitespace_count = {}
    infinite_count = {}

    for col in df.columns:
        if df[col].dtype == 'object':
            empty_string_count[col] = (df[col] == "").sum()
            whitespace_count[col] = df[col].apply(lambda x: isinstance(x, str) and x.strip() == "").sum()
        else:
            empty_string_count[col] = 0
            whitespace_count[col] = 0

        if pd.api.types.is_numeric_dtype(df[col]):
            infinite_count[col] = np.isinf(df[col]).sum()
        else:
            infinite_count[col] = 0

    report = pd.DataFrame({
        "Missing Count": df.isnull().sum(),
        "Missing %": df.isnull().mean() * 100,
        "NaN Count": df.isna().sum(),
        "Empty String Count": pd.Series(empty_string_count),
        "Whitespace Count": pd.Series(whitespace_count),
        "Infinite Count": pd.Series(infinite_count),
        "Unique Values": df.nunique(),
        "Data Type": df.dtypes
    }).sort_values(by="Missing %", ascending=False)

    return report

# =====================
# 2. DUPLICATES & CONSTANTS
# =====================
def detect_duplicates_and_constants(df):
    duplicate_rows = df.duplicated().sum()
    constant_features = [col for col in df.columns if df[col].nunique() == 1]
    return duplicate_rows, constant_features

# =====================
# 3. ALERTS & RECOMMENDATIONS
# =====================
def generate_alerts_and_recommendations(df, imbalance_threshold=0.9, skew_threshold=2.0, zero_threshold=0.5):
    alerts = []
    recommendations = []
    skewness_dict = {}

    # Constant columns
    _, constant_features = detect_duplicates_and_constants(df)
    for col in constant_features:
        alerts.append(f"{col} has constant value \"{df[col].iloc[0]}\" -> Constant")
        recommendations.append(f"Consider dropping constant column: {col}")

    # Categorical imbalance
    for col in df.select_dtypes(include=['object', 'category']).columns:
        top_freq = df[col].value_counts(normalize=True, dropna=False).max()
        if top_freq >= imbalance_threshold:
            alerts.append(f"{col} is highly imbalanced ({top_freq*100:.1f}%) -> Imbalance")
            recommendations.append(f"Consider resampling or grouping categories in: {col}")

    # Skewness in numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        col_data = df[col].dropna()
        if len(col_data) > 1:
            skew_val = skew(col_data)
            skewness_dict[col] = skew_val
            if abs(skew_val) >= skew_threshold:
                alerts.append(f"{col} is highly skewed (γ1 = {skew_val:.2f}) -> Skewed")
                recommendations.append(f"Consider log/sqrt/Box-Cox transform for: {col}")

    # Zero-dominated numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        zero_ratio = (df[col] == 0).mean()
        if zero_ratio >= zero_threshold:
            alerts.append(f"{col} has {zero_ratio*100:.1f}% zeros -> Zeros")
            recommendations.append(f"Consider imputing zeros or analyzing distribution for: {col}")

    # Missing values
    missing_cols = df.columns[df.isnull().any()]
    for col in missing_cols:
        if df[col].dtype == 'object':
            recommendations.append(f"Consider filling missing strings in {col} with mode or placeholder")
        else:
            recommendations.append(f"Consider filling missing numeric values in {col} with mean/median or using imputation")

    # Duplicates
    duplicate_rows, _ = detect_duplicates_and_constants(df)
    if duplicate_rows > 0:
        recommendations.append("Consider removing duplicate rows")

    return alerts, recommendations, skewness_dict

# =====================
# 4. PLOTS
# =====================
def plot_missing_values(df, stage='before'):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis")
    plt.title(f"Missing Value Heatmap ({stage})")
    plt.tight_layout()
    plt.show()

def plot_skewed_distributions(df, skewness_dict, skew_plot_limit=5):
    if skewness_dict:
        top_skewed = sorted(skewness_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:skew_plot_limit]
        fig, axes = plt.subplots(len(top_skewed), 1, figsize=(8, 4*len(top_skewed)))
        if len(top_skewed) == 1:
            axes = [axes]
        for ax, (col, skew_val) in zip(axes, top_skewed):
            sns.histplot(df[col].dropna(), kde=True, ax=ax, color='skyblue')
            ax.set_title(f"{col} - Skewness = {skew_val:.2f}", fontsize=12)
        plt.tight_layout()
        plt.show()

# =====================
# 5. EDA
# =====================
def perform_basic_eda(df, target_col=None):
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Correlation heatmap
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm')
        plt.title("Correlation Heatmap")
        plt.show()

    # Pairplots
    if len(numeric_cols) <= 5:
        sns.pairplot(df[numeric_cols])
        plt.show()

    # Target analysis
    if target_col and target_col in df.columns:
        plt.figure(figsize=(10, 6))
        if df[target_col].dtype in ['object', 'category']:
            sns.countplot(data=df, x=target_col)
            plt.title(f"Target Distribution: {target_col}")
        else:
            sns.histplot(df[target_col], kde=True)
            plt.title(f"Target Distribution: {target_col}")
        plt.show()