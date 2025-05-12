import pandas as pd
import os
import re

# Load the CSV file
df = pd.read_csv('/root/Datasets/2025_02_07/metadata.csv', delimiter='|', encoding='utf-8')

# Define the directory containing audio files
audio_dir = "/root/narakeet/text-to-speech-polling-api-python-example/data/merged_files_2/"

# Function to extract numeric index from filenames and add 1
def extract_number(filename):
    match = re.search(r"sentence_(\d+)\.wav", filename)
    return int(match.group(1)) - 2 if match else None  # Add 1 to the extracted number

# Get available audio file numbers (after adding 1)
audio_files = os.listdir(audio_dir)
audio_numbers = sorted({extract_number(file) for file in audio_files if extract_number(file) is not None})

# Filter sentences based on these updated indices
df_filtered = df.iloc[audio_numbers].copy()

# Create a new column for the file path
df_filtered["path"] = df_filtered.index.map(lambda x: f"sentence_{x + 2}.wav")  # Adjust back to match filenames

# Save the final metadata CSV
output_path = "/root/narakeet/text-to-speech-polling-api-python-example/data/metadata_2.csv"
df_filtered[["path", "sentence"]].to_csv(output_path, sep='|', index=False)

print(f"CSV file saved as {output_path}")
print(f"Total rows in final CSV: {len(df_filtered)}")