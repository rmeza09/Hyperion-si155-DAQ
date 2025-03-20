import pandas as pd

def calculate_sampling_rate(filename):
    # Step 1: Read the file while skipping the first 45 lines
    start_idx = 45  # Data starts on line 46 (0-based index)
    
    # Step 2: Read the table into a DataFrame
    try:
        df = pd.read_csv(filename, skiprows=start_idx, sep = '\s+', on_bad_lines="skip", engine="python")
    except Exception as e:
        print(f"❌ Pandas error: {e}")
        return

    # Debugging: Print column names to verify parsing
    print(f"Columns detected: {df.columns.tolist()}")

    # Step 3: Ensure the first two columns exist for timestamp parsing
    if len(df.columns) < 2:
        print("❌ Error: Expected at least two columns (Timestamp and Data). Check the separator.")
        return

    # Step 4: Extract timestamps and compute time elapsed
    df['Timestamp'] = pd.to_datetime(df.iloc[:, 0] + " " + df.iloc[:, 1], errors="coerce")  # Merge date & time
    df = df.dropna(subset=['Timestamp'])  # Remove rows with parsing issues

    if df.empty:
        print("❌ Error: No valid timestamps found. Check file format.")
        return

    start_time = df['Timestamp'].iloc[0]
    end_time = df['Timestamp'].iloc[-1]

    # Step 5: Compute elapsed time and sampling rate
    elapsed_time = (end_time - start_time).total_seconds()
    total_samples = len(df)

    sampling_rate = total_samples / elapsed_time if elapsed_time > 0 else 0

    # Print results
    print(f" Total Samples: {total_samples}")
    print(f" Start Time: {start_time}")
    print(f" End Time: {end_time}")
    print(f" Elapsed Time: {elapsed_time:.6f} seconds")
    print(f" Calculated Sampling Rate: {sampling_rate:.2f} Hz")

# Usage Example
filename = "./DATA/ENLIGHT/Peaks.20250320140346.txt"  # Replace with your actual file name
calculate_sampling_rate(filename)
