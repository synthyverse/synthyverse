import pandas as pd
import numpy as np
from scipy.stats import skew, boxcox

import os
import json
from datetime import datetime
from qualityCheck import calculate_basic_stats, detect_duplicates_and_constants
from qualityCheck import generate_alerts_and_recommendations, plot_missing_values, plot_skewed_distributions, perform_basic_eda
from clean import replace_empty_with_nan, replace_infinite_with_nan, impute_missing_values
from clean import remove_empty_and_constant_columns, correct_skewness, drop_duplicates, make_json_serializable

class preprocessing:
    # =====================
    # MAIN FUNCTION
    # =====================
    @staticmethod
    def data_quality_report(df, stage='before', output_prefix='data_quality',
                                    imbalance_threshold=0.9, skew_threshold=2.0, zero_threshold=0.5,
                                    plot_missing=True, plot_skew=True, skew_plot_limit=5,
                                    perform_eda=False, target_col=None, verbose=True):
        
        """
        Generate a detailed data quality report for a dataset.

        Parameters:
        -----------
        df : pandas.DataFrame
            The dataset you want to analyze.
        stage : str, default='before'
            Whether this report is before cleaning ('before') or after cleaning ('after').
        output_prefix : str, default='data_quality'
            Prefix used for naming output files if saved.
        imbalance_threshold : float, default=0.9
            If the most common class in a column is more frequent than this threshold,
            the column will be flagged for imbalance.
        skew_threshold : float, default=2.0
            Absolute skewness value above which a numeric column will be flagged as skewed.
        zero_threshold : float, default=0.5
            If more than this fraction of a column’s values are zero, it will be flagged.
        plot_missing : bool, default=True
            Whether to create a plot of missing values.
        plot_skew : bool, default=True
            Whether to create a plot of skewed numerical columns.
        skew_plot_limit : int, default=5
            Maximum number of skewed columns to plot.
        perform_eda : bool, default=False
            If True, perform basic exploratory data analysis (EDA) on the dataset.
        target_col : str or None, default=None
            Name of the target column for EDA purposes (used only if perform_eda=True).
        verbose : bool, default=True
            If True, prints all report details to the console.
            If False, only returns the report data without printing.

        Returns:
        --------
        report : pandas.DataFrame
            Summary statistics of the dataset.
        alerts : list of str
            Detected potential data quality issues.
        recommendations : list of str
            Suggested cleaning or preprocessing actions.
        """
        if verbose:
            print(f"Total samples (rows): {df.shape[0]}")
            print(f"Total features (columns): {df.shape[1]}")
            print(f"\n{'='*30}\n DATA QUALITY REPORT ({stage.upper()})\n{'='*30}")

        # Basic stats
        report = calculate_basic_stats(df)
        if verbose:
            print(report)

        # Duplicates & constants
        duplicate_rows, constant_features = detect_duplicates_and_constants(df)
        if verbose:
            print(f"\nDuplicate Rows: {duplicate_rows}")
            print(f"Constant Features: {constant_features if constant_features else 'None'}")

        # Alerts & Recommendations
        alerts, recommendations, skewness_dict = generate_alerts_and_recommendations(
            df, imbalance_threshold, skew_threshold, zero_threshold
        )

        if verbose:
            print(f"\n{'='*30}\n DATA ALERTS\n{'='*30}")
            if alerts:
                for alert in alerts:
                    print(alert)
            else:
                print("No major alerts detected.")

            print(f"\n{'='*30}\n CLEANING RECOMMENDATIONS\n{'='*30}")
            if recommendations:
                for rec in recommendations:
                    print(rec)
            else:
                print("No specific cleaning recommended.")

        # Plots
        if plot_missing:
            plot_missing_values(df, stage)
        if plot_skew:
            plot_skewed_distributions(df, skewness_dict, skew_plot_limit)

        # EDA
        if perform_eda:
            if verbose:
                print(f"\n{'='*30}\n EXPLORATORY DATA ANALYSIS\n{'='*30}")
            perform_basic_eda(df, target_col)

        return report, alerts, recommendations

    # =====================
    # 6. CLEANING FUNCTIONS
    # =====================
    # ==============================================================================


    # -------------------- Main Function --------------------
    @staticmethod
    def clean_and_impute(df, 
                            dataset_name,
                            num_strategy='mean', 
                            cat_strategy='mode', 
                            fill_constant='Unknown',
                            handle_skewness=False, 
                            skew_threshold=1.0, 
                            skew_method='log',
                            save_folder='cleaned_dataset',
                            remove_empty=True,
                            remove_constant=True):
        """
        Cleans and imputes missing/invalid values in the DataFrame, 
        with optional skewness correction.

        Parameters:
        - dataset_name: Name of the dataset (used in filenames)
        - num_strategy: Strategy for numeric imputation ('mean', 'median', 'zero')
        - cat_strategy: Strategy for categorical imputation ('mode', 'constant', 'drop')
        - fill_constant: Value used for constant fill strategy in categorical columns
        - handle_skewness: Whether to correct skewed numeric columns
        - skew_threshold: Skewness value above which correction will be applied
        - skew_method: Method to correct skew ('log', 'boxcox')
        - save_folder: Folder to save cleaned datasets
        - remove_empty: Whether to drop empty (all NaN) columns
        - remove_constant: Whether to drop constant (single unique value) columns
        """
        df_clean = df.copy()
        transformations = {}

        # Step 1: Replace empty strings with NaN
        df_clean = replace_empty_with_nan(df_clean)

        # Step 2: Replace infinite values with NaN
        df_clean = replace_infinite_with_nan(df_clean)

        # Step 3: Impute missing values
        df_clean = impute_missing_values(df_clean, num_strategy, cat_strategy, fill_constant)

        # Step 4: Remove empty and constant columns (if enabled)
        df_clean = remove_empty_and_constant_columns(df_clean, remove_empty, remove_constant)

        # Step 5: Handle skewness if required
        if handle_skewness:
            df_clean, transformations = correct_skewness(df_clean, skew_threshold, skew_method)

        # Step 6: Drop duplicates
        df_clean = drop_duplicates(df_clean)

        # Step 7: Create folder if it doesn't exist
        os.makedirs(save_folder, exist_ok=True)

        # Step 8: Add timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Step 9: Save cleaned dataset
        clean_file_path = os.path.join(save_folder, f'{dataset_name}_cleaned_{timestamp}.csv')
        df_clean.to_csv(clean_file_path, index=False)
        print(f"Cleaned dataset saved to: {clean_file_path}")

        # Step 10: Save transformations only if not empty
        if transformations:
            transformations_serializable = make_json_serializable(transformations)
            transformations_file_path = os.path.join(save_folder, f'{dataset_name}_transformations_{timestamp}.json')
            with open(transformations_file_path, 'w') as f:
                json.dump(transformations_serializable, f, indent=4)
            print(f"Transformations saved to: {transformations_file_path}")
        else:
            print("No transformations applied — no transformation file saved.")

        return df_clean, transformations
    # ==============================================================================
    # REVERSE TRANSFORMATIONS
    # ==============================================================================
    @staticmethod
    def reverse_transformations(df, transformations):
        """
        Reverse transformations applied for skewness correction.

        Parameters:
            df (pd.DataFrame): Transformed DataFrame
            transformations (dict): Transformations applied

        Returns:
            pd.DataFrame: DataFrame restored to original scale
        """
        df_reversed = df.copy()

        for col, params in transformations.items():
            if params[0] == 'log':
                shift = params[1]
                df_reversed[col] = np.expm1(df_reversed[col]) - shift
            elif params[0] == 'boxcox':
                from scipy.special import inv_boxcox
                shift = params[1]
                lam = params[2]
                df_reversed[col] = inv_boxcox(df_reversed[col], lam) - shift

        return df_reversed

    # ==============================================================================
    # SAVE DATA QUALITY COMPARISON
    # ==============================================================================
    @staticmethod
    def data_quality_comparison(report_before, alerts_before, recs_before,
                                    report_after, alerts_after, recs_after,
                                    dataset_name="Dataset"):
        """
        Save before/after data quality comparison into a single Excel file.

        Parameters:
        - report_before, alerts_before, recs_before: Output from generate_data_quality_report (before cleaning)
        - report_after, alerts_after, recs_after: Output from generate_data_quality_report (after cleaning)
        - dataset_name: Used in output file naming
        """

        # Ensure DataFrames for column stats
        comparison_df = report_before.add_suffix("_Before").merge(
            report_after.add_suffix("_After"), 
            left_index=True, right_index=True, how="outer"
        )

        # Alerts comparison
        alerts_df = pd.DataFrame({
            "Alerts_Before": alerts_before + [""] * max(0, len(alerts_after) - len(alerts_before)),
            "Alerts_After": alerts_after + [""] * max(0, len(alerts_before) - len(alerts_after))
        })

        # Recommendations comparison
        recs_df = pd.DataFrame({
            "Recommendations_Before": recs_before + [""] * max(0, len(recs_after) - len(recs_before)),
            "Recommendations_After": recs_after + [""] * max(0, len(recs_before) - len(recs_after))
        })

        # Save all to one Excel file
        output_path = f"{dataset_name}_DataQuality_Comparison.xlsx"
        with pd.ExcelWriter(output_path) as writer:
            comparison_df.to_excel(writer, sheet_name="Column_Stats")
            alerts_df.to_excel(writer, sheet_name="Alerts", index=False)
            recs_df.to_excel(writer, sheet_name="Recommendations", index=False)

        print(f"Data quality comparison report saved to: {output_path}")
        return output_path
