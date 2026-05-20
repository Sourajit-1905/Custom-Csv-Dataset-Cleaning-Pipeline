import pandas as pd
import numpy as np


def load_dataset(file_path: str) -> pd.DataFrame:
    """Loads a CSV dataset, treating common missing value strings as NaN."""

    missing_identifiers = ["N/A", "n/a", "NA", "na", "Missing", "missing", "None", "null", "NULL", " "]
    try:
        df = pd.read_csv(file_path, na_values=missing_identifiers)
        return df
    
    except FileNotFoundError:
        print(f"File not found at {file_path}")
        raise

    except Exception as e:
        print(f"{e}")
        raise

    finally:
        print("The next step is to Handle Duplicate Rows and Columns")

    

def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Identifies and drops exact duplicate rows and columns from the dataset."""

    try:
        # Handle Duplicate Rows
        initial_rows = df.shape[0] # df.shape() returns a typle of (rows, columns)
        df_cleaned = df.drop_duplicates().reset_index(drop=True)
        
        # Handle Duplicate Columns
        df_cleaned = df_cleaned.T.drop_duplicates(keep="first").T.convert_dtypes()
        
        return df_cleaned
    
    except Exception as e:
        print(f"{e}")
        raise

    finally:
        print("The next step is to Handle Missing Values")



def handle_missing_values(df: pd.DataFrame, strategy: str = "auto") -> pd.DataFrame:
    """
    Handles missing values dynamically based on the column data type.
    
    Strategies available:
    - 'auto': Fills numeric columns with their median, categorical with their mode.
    - 'drop': Drops any rows containing missing values.
    """

    df_cleaned = df.copy()

    if strategy == "auto":
        for column in df_cleaned.columns:
            missing_count = df_cleaned[column].isna().sum()

            if missing_count > 0:
                # Check if column is numeric
                if pd.api.types.is_numeric_dtype(df_cleaned[column]):
                    median_value = df_cleaned[column].median()
                    df_cleaned[column] = df_cleaned[column].fillna(median_value)

                else:
                    # Categorical or textual column                   
                    if not df_cleaned[column].mode().empty:
                        mode_value = df_cleaned[column].mode()[0]
                        df_cleaned[column] = df_cleaned[column].fillna(mode_value)

                    else:
                        df_cleaned[column] = df_cleaned[column].fillna("Unknown")
                        
    elif strategy == "drop":
        df_cleaned = df_cleaned.dropna().reset_index(drop=True)

    else:
        print("Invalid Missing Value Handling Mode")


    print("The next step is to Cast Data Types")

    return df_cleaned



def cast_data_types(df: pd.DataFrame, date_columns: list = None, boolean_columns: list = None) -> pd.DataFrame:
    """
    Cleans structural inconsistencies, standardizes strings, and casts 
    data types safely using dynamic evaluation.
    """
    df_cleaned = df.copy()

    # 1. Strip whitespace from object columns and convert binary indicators to true booleans
    for column in df_cleaned.columns:
        if pd.api.types.is_object_dtype(df_cleaned[column]):
            df_cleaned[column] = df_cleaned[column].astype(str).str.strip()

    # 2. Process Boolean Conversions
    if boolean_columns:

        bool_mapping = {
            "true": True, "yes": True, "1": True, "1.0": True, "y": True, "t": True,
            "false": False, "no": False, "0": False, "0.0": False, "n": False, "f": False
        }

        for col in boolean_columns:

            if col in df_cleaned.columns:
                # Convert strings to lowercase to catch mixed configurations ('TRUE', 'yes', etc)
                string_series = df_cleaned[col].astype(str).str.lower()
                df_cleaned[col] = string_series.map(bool_mapping).fillna(False).astype(bool)

    # 3. Process Datetime Conversions
    if date_columns:
        for col in date_columns:

            if col in df_cleaned.columns:
                # 'mixed' format evaluation ensures dates like YYYY-MM-DD and DD-MM-YYYY compile uniformly
                df_cleaned[col] = pd.to_datetime(df_cleaned[col], format='mixed', errors='coerce')
                
    print("The next step is to Handle Outliers using Inter Queartile Ranges")

    return df_cleaned



def handle_outliers_iqr(df: pd.DataFrame, numeric_columns: list = None, strategy: str = "cap") -> pd.DataFrame:
    """
    Detects outliers across specified numeric features using the Interquartile Range (IQR) method.
    
    Strategies available:
    - 'cap': Caps values beyond boundaries to the 1.5 * IQR limit (Winsorization principle).
    - 'drop': Removes rows containing extreme outlier evaluations.
    """

    df_cleaned = df.copy()
    
    if not numeric_columns:
        numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
        
    for col in numeric_columns:
        if col in df_cleaned.columns:

            q1 = df_cleaned[col].quantile(0.25)
            q3 = df_cleaned[col].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            
            outliers_mask = (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)
            outliers_count = outliers_mask.sum()
            
            if outliers_count > 0:

                if strategy == "cap":
                    df_cleaned[col] = np.clip(df_cleaned[col], lower_bound, upper_bound)

                elif strategy == "drop":
                    df_cleaned = df_cleaned[~outliers_mask]
                    
    df_cleaned = df_cleaned.reset_index(drop=True)

    return df_cleaned



if __name__ == "__main__":

    FILE_PATH = "sample1.csv"
    OUTPUT_FILE_PATH = "output_sample1.csv"
    
    # The pipeline handles basic string anomalies automatically, but datetime/boolean require mapping guidance
    DATE_COLS = ["Release_Date", "date", "created_at"] 
    BOOL_COLS = ["Is_Active", "status", "verified"]
    
    print("STARTING AUTOMATED DATA CLEANING PIPELINE\n")
    
    try:
        raw_df = load_dataset(FILE_PATH)
        print("\n--- RAW DATA PREVIEW ---")
        print(raw_df.head(5))
        print(raw_df.dtypes)
        print("-" * 30 + "\n")


        # Clean Duplicates
        df_step1 = handle_duplicates(raw_df)
        
        # Standardize Structuring & Cast Data Types
        df_step2 = cast_data_types(df_step1, date_columns=DATE_COLS, boolean_columns=BOOL_COLS)
        
        # Process Outliers via IQR Rules
        df_step3 = handle_outliers_iqr(df_step2, strategy="cap")
        
        # Resolve Missing Values/NaN entries
        final_clean_df = handle_missing_values(df_step3, strategy="auto")
        
        print("\n=== CLEANING COMPLETE ===")
        print("\n--- FINAL DATA PREVIEW ---")
        print(final_clean_df.head(5))
        print(final_clean_df.dtypes)
        print(f"\nFinal Dataset Size: {final_clean_df.shape[0]} rows, {final_clean_df.shape[1]} columns")
        
        # Export clean output back to your directory
        final_clean_df.to_csv(OUTPUT_FILE_PATH, index=False)
        print(f"\nCleaned dataset exported as: '{OUTPUT_FILE_PATH}'")
        
    except FileNotFoundError:
        print("\nDataset Not in same directory")
        
    except Exception as pipeline_error:
        print(f"\nCleaning Pipeline execution halted: {pipeline_error}")


