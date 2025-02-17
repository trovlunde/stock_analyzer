import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FuncFormatter
import numpy as np

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# List of CSV files to process with exact filenames
csv_files = [
    os.path.join(current_dir, 'data', 'creviceCorrosionPotential.csv'),
    os.path.join(current_dir, 'data', 'pittingPotential.csv'),
    os.path.join(current_dir, 'data', 'repassivationPotential.csv')
]

# Print the paths to verify
for file in csv_files:
    print(f"Looking for file: {file}")
    print(f"File exists: {os.path.exists(file)}")


def heatmap_function(csv_file, required_columns, variable):
    """
    Create heatmaps from CSV data
    """
    try:
        df = pd.read_csv(csv_file)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_file, encoding='latin1')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(csv_file, encoding='iso-8859-1')
            except UnicodeDecodeError:
                df = pd.read_csv(csv_file, encoding='cp1252')

    print(f"Reading file: {csv_file}")
    df.columns = df.columns.str.strip()

    # Drop rows with missing values in critical columns
    df = df.dropna(subset=required_columns)

    # Ensure material class is treated as a category
    df['Material class'] = df['Material class'].astype('category')

    # Ensure numeric columns are numeric
    numeric_cols = [required_columns[0], required_columns[1],
                    required_columns[2]]  # Epit, Test Temp, [Cl-]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Sort the dataframe by [Cl-] concentration
    df = df.sort_values(by=required_columns[2])

    print("\nPivot table preview:")
    pivot_data_temp = df.pivot_table(
        index=required_columns[1],  # Temperature
        columns=required_columns[3],  # Material class
        values=required_columns[0],  # Epit
        aggfunc='mean'
    ).fillna(np.nan)

    if pivot_data_temp.empty:
        print("\nError: Pivot table is empty!")
        return

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # First heatmap (Temperature)
    sns.heatmap(pivot_data_temp, annot=True, cmap='YlGnBu',
                cbar_kws={'label': variable}, ax=ax1)
    ax1.set_title(
        f"Heatmap of {variable} vs Test Temp\nfor Different Material Classes")
    ax1.set_xlabel("Material Class")
    ax1.set_ylabel("Test Temperature (°C)")

    # Create second pivot table for chloride concentration
    pivot_data_cl = df.pivot_table(
        index=required_columns[2],  # [Cl-]
        columns=required_columns[3],  # Material class
        values=required_columns[0],  # Epit
        aggfunc='mean'
    ).fillna(np.nan)

    # Sort the chloride concentration index
    pivot_data_cl = pivot_data_cl.sort_index()

    # Second heatmap ([Cl-])
    sns.heatmap(pivot_data_cl, annot=True, cmap='coolwarm',
                cbar_kws={'label': variable}, ax=ax2,
                yticklabels=[f'{x:.3f}' for x in pivot_data_cl.index])  # Use actual [Cl-] values
    ax2.set_title(
        f"Heatmap of {variable} vs Chloride Ion\nConcentration for Different Material Classes")
    ax2.set_xlabel("Material class")
    ax2.set_ylabel("[Cl-] (M)")

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

    print("\nChloride concentration values:")
    print(df[required_columns[2]].unique())

    return


heatmap_function(csv_files[1], [
                 'Epit, mV (SCE)', 'Test Temp.', '[Cl-]', 'Material class'], "Epit")

# heatmap_function(csv_files[2], ['Ave. Erp', 'Test Temp', '[Cl-]', 'Material class'], "Erep")

# heatmap_function(csv_files[0], ['Ecrev, mV (SCE)', 'Test Temp.','[Cl-]','Unnamed: 28'], 'Ecrev')
