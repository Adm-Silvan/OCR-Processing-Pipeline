import os
import json
import pandas as pd

def extract_doc_energy_info(file_path, doc_id, codecarbon_csv_path, token_count_csv_path, output_csv_path):
    """
    Given a document ID, extract energy, duration, gpu power, emissions from CodeCarbon CSV,
    and token count from a token count CSV. Output a row with these fields plus the document ID 
    to a new CSV file.

    Parameters:
    - doc_id: The document ID to search for in CodeCarbon CSV
    - codecarbon_csv_path: Path to the CodeCarbon output CSV file
    - token_count_csv_path: Path to the token count CSV file (columns: FilePath, TokenCount)
    - docid_to_filepath: dict mapping doc_id to file path (matching token count CSV file paths)
    - output_csv_path: Path to save the output CSV with the combined info

    Returns:
    None (writes to output_csv_path)
    """

    # Load CodeCarbon CSV
    cc_df = pd.read_csv(codecarbon_csv_path)
    
    # Try to find the row with matching doc_id
    row = cc_df[cc_df['doc_id'] == int(doc_id)]
    if row.empty:
        print(f"Doc ID {doc_id} not found in CodeCarbon CSV. Skipping.")
        return

    # Extract fields from the first matching row (assuming only one needed)
    energy_consumed = row.iloc[0].get('energy_consumed', None)
    duration = row.iloc[0].get('duration', None)
    gpu_power = row.iloc[0].get('gpu_power', None)
    emissions = row.iloc[0].get('emissions', None)

    # Load token counts CSV
    token_df = pd.read_csv(token_count_csv_path)


    # Try to find token count row by file path
    token_row = token_df[token_df['FilePath'] == file_path]
    if token_row.empty:
        print(f"Token count not found for file path {file_path}.")
        token_count = None
    else:
        token_count = int(token_row.iloc[0]['TokenCount'])

    # Prepare output dictionary
    output_data = {
        'doc_id': doc_id,
        'energy_consumed': energy_consumed,
        'duration': duration,
        'gpu_power': gpu_power,
        'emissions': emissions,
        'token_count': token_count
    }

    # Write output as a single-row CSV (append mode if the file exists)
    output_df = pd.DataFrame([output_data])
    try:
        # Append if file exists else write with header
        with open(output_csv_path, 'a', encoding='utf-8', newline='') as f:
            output_df.to_csv(f, header=f.tell() == 0, index=False)
        print(f"Output written for doc ID {doc_id}")
    except Exception as e:
        print(f"Error writing output for doc ID {doc_id}: {e}")

LOGS_ROOT = "C:/Users/Silvan/Data/Logs/"
PATH = "C:/Users/Silvan/Data/OCR_Protocols/"

for year in range(1904,1973):
    for month in range(1,13):
        with open(PATH+f"Manifests/{year}-{month}.json", "r", encoding="utf-8") as file:
            data = json.load(file)
        for date, values in data.items():
            doc_id = values[0]
            signature = values[1]
            print(f"Date: {date}, ID: {id},Signature {signature}")
            day = date.split(".")[0]
            if len(str(month)) < 2: month = "0"+str(month)
            file_path = PATH+f"{year}/{month}/{year}-{month}-{day}.txt"
            if os.path.exists(file_path):
                 print(file_path)
                 codecarbon_csv_path = LOGS_ROOT+"emissions_per_doc.csv"
                 token_count_csv_path = LOGS_ROOT+"token_counts.csv"
                 output_csv_path = LOGS_ROOT+"aggregate_log.csv"
                 extract_doc_energy_info(file_path, doc_id, codecarbon_csv_path, token_count_csv_path, output_csv_path)
            else:
                print(PATH+f"{year}/{month}/{year}-{month}-{day}.txt is missing")
                with open("missing_files.txt", "a") as file:
                    file.write(date + " " + values[0]+ " " + values[1] + "\n")