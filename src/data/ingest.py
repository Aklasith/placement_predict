import os
import pandas as pd

def load_data(file_path):
    """
    Load the placement dataset from a CSV file.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    print(f"Executing secure data extraction from: {file_path}")

    df = pd.read_csv(file_path)

    required_columns = [
    'branch',
    'college_tier',
    'cgpa',
    'backlogs',
    'coding_skill_score',
    'communication_skill_score',
    'internships_count',
    'projects_count',
    'placement_status',
    'salary_package_lpa'
]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df

if __name__ == "__main__":
    DATA_PATH = os.path.join("src", "data", "raw_placement_data.csv")

    try:
        raw_data = load_data(DATA_PATH)
        print("Data ingestion completed successfully!")
        print(raw_data.head())

    except Exception as e:
        print(f"Ingestion lifecycle termination: {str(e)}")