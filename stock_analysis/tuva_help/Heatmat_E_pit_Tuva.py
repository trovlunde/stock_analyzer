import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

csv_file = 'Crevice Corrosion Potential.csv'
df = pd.read_csv(csv_file)
df.columns = df.columns.str.strip()
print("Columns = ", df.columns)

def heatmap_function(csv_file, required_columns, variable):

    # Load the CSV file
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()
    print("Columns = ", df.columns)

    # Drop rows with missing values in critical columns
    #df = df.dropna(subset=required_columns)

    # Ensure columns are numeric
    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    #df = df.dropna(subset=required_columns)

    print("DataFrame shape after dropna:", df.shape)

    print("Missing values in each column:", df.isna().sum())


    # Pivot table where the rows are 'Material class' and columns are combinations of 'Epit, mV (SCE)' and 'Test Temp.'
    # This aggregates the data for the heatmap.
    pivot_data_with_na = df.pivot_table(
        index=required_columns[1],
        columns=required_columns[3],
        values=required_columns[0],
    ).fillna(np.nan)  # Replace NA with -1 or another marker

    # Heatmap using the relationship between 'Epit, mV (SCE)', 'Test Temp.', and '[Cl-]'
    pivot_data2_with_na = df.pivot_table(
        index=required_columns[2],
        columns=required_columns[3],
        values=required_columns[0],
    ).fillna(np.nan)  # Replace NA with -1 or another marker

    #pivot_data2_with_na = pivot_data2_with_na.sort_index()

    # Create the heatmap using seaborn, Temperature
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_data_with_na, annot=True, cmap='YlGnBu', cbar_kws={'label': variable})
    plt.title("Heatmap of {variable} vs Test Temp for Different Material Classes incl. missing values")
    plt.xlabel("Material Class")
    plt.ylabel("Test Temperature (°C)")
    plt.show()

    # Create the second heatmap for [Cl-]
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_data2_with_na, annot=True, cmap='coolwarm', cbar_kws={'label': variable})

    #plt.yticks(ticks=range(len(pivot_data2_with_na.index)), labels=pivot_data2_with_na.index)

    plt.title(f"Heatmap of {variable} vs Chloride Ion Concentration for Different Material Classes incl. missing values")
    plt.xlabel("Material class")
    plt.ylabel("[Cl-] (M)")
    plt.ylim(0, len(pivot_data2_with_na.index))
    # Use FuncFormatter to format the y-axis with three decimals
    def format_ticks(x, pos):
        return f'{x:.3f}'
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_ticks))
    plt.show()

    print(df[required_columns[2]].unique())
    print(df[required_columns[0]].unique())

    return

heatmap_function('Pitting Potential.csv', ['Epit, mV (SCE)', 'Test Temp.', '[Cl-]', 'Material class'], "Epit")

#heatmap_function('Repassivation Potential.csv', ['Ave. Erp', 'Test Temp', '[Cl-]', 'Material class'], "Erep")

#heatmap_function('Crevice Corrosion Potential.csv', ['Ecrev, mV (SCE)', 'Test Temp.','[Cl-]','Unnamed: 28'], 'Ecrev')