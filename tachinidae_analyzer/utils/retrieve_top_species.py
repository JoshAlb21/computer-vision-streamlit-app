import pandas as pd

def categorize_top_species(dataframe, column_name='Species', top_n=5, nan_equals_others=True):
    """
    Categorizes the most common species in a dataframe column and sets the rest to 'Others'.
    
    Parameters:
    - dataframe (pd.DataFrame): Input dataframe
    - column_name (str): Name of the column to categorize
    - top_n (int): Number of top values to keep, rest will be set to 'Others'
    
    Returns:
    - pd.DataFrame: Dataframe with the column updated
    """
    
    # Count the values in the specified column
    value_counts = dataframe[column_name].value_counts()
    
    # Keep the top_n species
    top_values = value_counts.head(top_n).index.tolist()
    
    # Replace values that are not in the top_n with 'Others'
    dataframe[column_name] = dataframe[column_name].where(dataframe[column_name].isin(top_values), 'Others')

    if nan_equals_others:
        dataframe[column_name] = dataframe[column_name].fillna('Others')
    
    return dataframe