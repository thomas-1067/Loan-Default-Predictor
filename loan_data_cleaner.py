def clean_loan_data(df, cap_outliers=True, cap_quantile=0.99):
    """
    Cleans credit risk data by handling duplicates, invalid values, and missing data.
    
    Parameters:
        df (pd.DataFrame): Raw input dataframe
        cap_outliers (bool): Whether to cap extreme values
        cap_quantile (float): Quantile threshold for capping numerical features (default 0.99)

    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    df = df.copy()

    # Drop duplicate rows
    df.drop_duplicates(inplace=True)

    # Remove any age < 18
    if 'age' in df.columns:
        df = df[df['age'] >= 18]

    # Fill missing values
    if 'MonthlyIncome' in df.columns:
        df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())

    if 'NumberOfDependents' in df.columns:
        df['NumberOfDependents'] = df['NumberOfDependents'].fillna(df['NumberOfDependents'].median())

    # Cap outliers using quantiles
    if cap_outliers:
        numeric_cols = df.select_dtypes(include=[float, int]).columns
        for col in numeric_cols:
            upper = df[col].quantile(cap_quantile)
            df[col] = df[col].clip(upper=upper)

    return df
