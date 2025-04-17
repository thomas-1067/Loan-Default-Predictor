def check_dataset(df, name="Dataset"):
    """
    Check for basic data integrity issues in a DataFrame.
    Prints a summary of findings.

    Parameters:
        df (pd.DataFrame): Dataset to check
        name (str): Optional name of dataset for printing context

    Returns:
        dict: Summary of findings
    """
    summary = {}

    print(f"Inspection Report for {name}")
    print("-" * 50)

    # 1. Unnamed columns
    unnamed_cols = [col for col in df.columns if col.startswith("Unnamed")]
    if unnamed_cols:
        print(f"❌ Unnamed columns found and removed: {unnamed_cols}")
        df.drop(columns=unnamed_cols, inplace=True)
        summary["unnamed_cols_removed"] = unnamed_cols
    else:
        print("✅ No unnamed columns.")

    # 2. Duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"❌ {duplicate_count} duplicate rows found.")
        summary["duplicate_rows"] = duplicate_count
    else:
        print("✅ No duplicate rows.")

    # 3. Constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    if constant_cols:
        print(f"⚠️ Constant columns detected: {constant_cols}")
        summary["constant_columns"] = constant_cols
    else:
        print("✅ No constant columns.")

    # 4. Mostly zero columns
    mostly_zero = [col for col in df.columns if (df[col] == 0).mean() > 0.95]
    if mostly_zero:
        print(f"⚠️ Columns with >95% zeros: {mostly_zero}")
        summary["mostly_zero_columns"] = mostly_zero
    else:
        print("✅ No columns with >95% zeros.")

    # 5. Non-numeric columns
    non_numeric = df.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric:
        print(f"⚠️ Non-numeric columns: {non_numeric}")
        summary["non_numeric_columns"] = non_numeric
    else:
        print("✅ All columns are numeric.")

    # 6. Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print("⚠️ Missing values detected:")
        print(missing)
        summary["missing_values"] = missing.to_dict()
    else:
        print("✅ No missing values.")

    # 7. Invalid value checks (domain-specific)
    if 'age' in df.columns:
        invalid_ages = (df['age'] < 18).sum()
        if invalid_ages > 0:
            print(f"❌ {invalid_ages} rows with age < 18")
            summary["invalid_ages"] = invalid_ages
        else:
            print("✅ All age values >= 18.")

    print("-" * 50)
    return summary
