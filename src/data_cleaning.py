import pandas as pd

def remove_columns(df: pd.DataFrame, cols_to_remove: list) -> pd.DataFrame:
    """
    Removes specified columns from DataFrame if they exist.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to clean.
    cols_to_remove (list): List of column names to drop.

    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    existing_cols = [col for col in cols_to_remove if col in df.columns]
    return df.drop(columns=existing_cols)
