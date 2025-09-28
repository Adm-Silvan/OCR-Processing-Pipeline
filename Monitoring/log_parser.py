import pandas as pd
from datetime import datetime
import re

def parse_log_file(log_file_path):
    """
    Parses the log file to extract timestamps and chunk ID roots.
    Returns a list of tuples: (timestamp_datetime, chunk_id_root)
    Captures only lines with "Stored triples for chunk" and extracts chunk root (before '/').
    """
    pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*Stored triples for chunk (\d+)/\d+')
    entries = []
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                timestamp_str = match.group(1)
                chunk_root = match.group(2)
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                entries.append((timestamp, chunk_root))
    return entries

def find_chunk_id_transitions(log_entries):
    """
    From the list of (timestamp, chunk_root), find timestamps where chunk_root changes.
    Returns a list of tuples: (transition_timestamp, new_chunk_root)
    Includes the first entry always as a start.
    """
    transitions = []
    last_root = None
    for timestamp, chunk_root in log_entries:
        if chunk_root != last_root:
            transitions.append((timestamp, chunk_root))
            last_root = chunk_root
    return transitions

def match_transitions_to_csv(df, csv_ts_col, transitions):
    """
    For each transition (timestamp, chunk_root), find closest timestamp in the csv df (csv_ts_col).
    Return a mapping of csv index to chunk_root to assign doc_id.
    """
    csv_timestamps = pd.to_datetime(df[csv_ts_col])
    mapping = {}

    for trans_time, chunk_root in transitions:
        # Find index of the closest timestamp in CSV to transition time
        time_diffs = abs(csv_timestamps - trans_time)
        closest_idx = time_diffs.idxmin()
        mapping[closest_idx] = chunk_root

    return mapping

def propagate_doc_ids(df, mapping):
    """
    Given a dataframe and a mapping of csv indices to chunk_root doc_ids,
    assign doc_id to rows in a non-overlapping manner:
    - Assign each doc_id starting at the mapped index
      and ending just before the next mapped index.
    This prevents the same chunk id being assigned multiple times to overlapping rows.
    """
    df = df.copy()
    df['doc_id'] = None

    sorted_indices = sorted(mapping.keys())
    n = len(sorted_indices)
    for i, start_idx in enumerate(sorted_indices):
        doc_id = mapping[start_idx]
        # Define end index: just before next start index, or end of dataframe
        end_idx = sorted_indices[i+1] - 1 if i + 1 < n else len(df) - 1
        df.loc[start_idx:end_idx, 'doc_id'] = doc_id

    return df

def link_codecarbon_log(codecarbon_csv, log_file, output_csv=None):
    """
    Main function to link CodeCarbon CSV data with chunk roots from a log file.
    """
    df = pd.read_csv(codecarbon_csv)
    # Adjust timestamp column name if needed; here assumed 'timestamp'
    timestamp_col = 'timestamp'

    # Parse log file
    log_entries = parse_log_file(log_file)
    if not log_entries:
        raise ValueError("No chunk entries found in log file.")

    # Detect transitions in chunk roots
    transitions = find_chunk_id_transitions(log_entries)

    # Match transitions to CSV timestamps
    mapping = match_transitions_to_csv(df, timestamp_col, transitions)

    # Propagate doc_id for all CSV rows without overlap
    df_with_doc_id = propagate_doc_ids(df, mapping)

    # Save result if output path provided
    if output_csv:
        df_with_doc_id.to_csv(output_csv, index=False)

    return df_with_doc_id

if __name__ == '__main__':
    df_result = link_codecarbon_log("C:/Users/Silvan/Data/Logs/emissions.csv", "C:/Users/Silvan/Data/Logs/log.log", "emissions_per_doc.csv")
    print("Processed CodeCarbon CSV with appended doc_id column.")
