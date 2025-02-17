import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Function to combine corrosion data


def combined_data(file_path, sheet_column_mapping, output_csv_path):
    # Initialize an empty DataFrame for combined data
    combined_data = pd.DataFrame()

    # Iterate through each sheet and its corresponding column mapping
    for sheet_name, column_map in sheet_column_mapping.items():
        # Extract original column names from column_map
        original_columns = list(column_map.keys())

        # Read the sheet into a DataFrame with relevant columns
        sheet_data = pd.read_excel(
            file_path, sheet_name=sheet_name, usecols=original_columns)

        # Rename the columns to standardized names
        sheet_data.rename(columns=column_map, inplace=True)

        # Add a new column for the corrosion type (based on the specific sheet name)
        # Adds the sheet name as a column
        sheet_data['Sheet Name'] = sheet_name

        # Append the sheet data to the combined DataFrame
        combined_data = pd.concat(
            [combined_data, sheet_data], ignore_index=True)

    # Skip the first row of the combined DataFrame (index 0)
    combined_data = combined_data.iloc[1:].reset_index(drop=True)

    # Save the combined DataFrame to a CSV file
    combined_data.to_csv(output_csv_path, index=False)
    print(f"Combined data saved to {output_csv_path}")


# Example usage:
file_path = "CRA_database.xlsx"
sheet_column_mapping = {
    "Repassivation Potential": {
        "Ave. Erp": "Corrosion Potential",
        "[Cl-]": "[Cl-]",
        "Test Temp": "Test Temp",
        "Material class": "Material Class"
    },
    "Pitting Potential": {
        "Epit, mV (SCE)": "Corrosion Potential",
        "[Cl-]": "[Cl-]",
        "Test Temp.": "Test Temp",
        "Material class": "Material Class"
    },
    "Crevice Corrosion Potential": {
        "Ecrev, mV (SCE)": "Corrosion Potential",
        "[Cl-]": "[Cl-]",
        "Test Temp.": "Test Temp",
        "Unnamed: 28": "Material Class"
    }
}

output_csv_path = "combined_data.csv"

combined_data(file_path, sheet_column_mapping, output_csv_path)

# Read the combined CSV file to check the result
combined_df = pd.read_csv(output_csv_path)

# Print the entire DataFrame
print('Combined dataframe:\n', combined_df)


# Making the model:

def regression_model():
    # Load the data
    df = pd.read_csv(output_csv_path)

    # Check if corrosion potential has valid value (numeric and no '>' character)
    def is_corrosion_potential_valid(value):
        try:
            # Check if the value contains '>' or is a non-numeric string
            if isinstance(value, str) and '>' in value:
                return False
            float(value)  # Try converting to float to check if it's numeric
            return True
        except ValueError:
            return False

    # Separate valid and missing corrosion potential data
    train_data = df[df['Corrosion Potential'].apply(
        is_corrosion_potential_valid)]  # Valid data
    missing_data = df[~df['Corrosion Potential'].apply(
        is_corrosion_potential_valid)]  # Invalid data

    # Features and target for training
    train_data = train_data.dropna(subset=['Corrosion Potential'])
    x_train = train_data[['Test Temp', '[Cl-]']]  # Features
    y_train = train_data['Corrosion Potential']  # Target

    # Split the data into training and testing datasets with stratification based on 'Material class'
    x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42, stratify=train_data['Material Class'])

    # Make and train the Random Forest model
    model = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42)
    model.fit(x_train_split, y_train_split)

    # Predict the missing values using the trained model
    # Predictor for rows with missing corrosion potential
    x_missing = missing_data[['Test Temp', '[Cl-]']]
    # Get predictions for missing rows
    predicted_values = model.predict(x_missing)

    # Assign the predicted values to the corrosion potential column for these rows
    df.loc[missing_data.index, 'Corrosion Potential'] = predicted_values

    # Evaluate the model on the test set
    y_pred = model.predict(x_test_split)
    print("Predictions on test set:", y_pred)

    # Print the updated DataFrame
    print("Updated Data with Predicted Values for Missing Corrosion Potential:")
    print(df)

    # Calculate performance metrics
    mae = mean_absolute_error(y_test_split, y_pred)
    mse = mean_squared_error(y_test_split, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_split, y_pred)

    # Print the performance metrics
    print("Model Performance on Test Set:")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-squared (R²): {r2}")

    print('Y test split', np.array(y_test_split))
    print('Length y test split', len(y_test_split))
    print('Y pred', np.array(y_pred))
    print('Length y pred', len(y_pred))

    # Plot Actual vs. Predicted values
    plt.figure(figsize=(8, 6))
    y_test_split = pd.to_numeric(y_test_split, errors='coerce')
    plt.scatter(np.array(y_test_split), np.array(y_pred), color='blue')
    plt.plot([y_test_split.min(), y_test_split.max()], [y_test_split.min(), y_test_split.max()], color='red',
             linestyle='--')
    plt.xlabel('Actual Corrosion Potential')
    plt.ylabel('Predicted Corrosion Potential')
    plt.title('Actual vs. Predicted Corrosion Potential')
    plt.show()

    # Plot Residuals- difference actual value and predicted value, residuals = errors
    y_test_split = pd.to_numeric(y_test_split)
    y_pred = pd.to_numeric(y_pred)
    residuals = y_test_split - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, color='blue')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted corrosion potential')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.show()

    # Feature Importance Plot (for RandomForestRegressor)
    # How important each feature is in making predictions
    plt.figure(figsize=(8, 6))
    plt.barh(x_train.columns, model.feature_importances_)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance Plot')
    plt.show()

    return


regression_model()
