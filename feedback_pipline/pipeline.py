import openai
import pandas as pd
import os
import re
import glob
from narakeet_api import AudioAPI
import datetime
import shutil
import subprocess
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
from utils import backup_models,clean_urdu_text, generate_sentences_for_word, update_global_error_feedback_from_df,move_checkpoints_and_config


import os
import json
import pandas as pd

print("Maleeeeeeeeeeee")

# Paths
feedback_csv_path = "/root/piper/src/python/streamlit_output/model_error_feedback.csv"
last_processed_json = "/root/piper/src/python/feedback_pipline/last_processed_run_male.json"
# Source folder to backup
src_dir = "/root/piper/trained_models/Narakeet_base_10k_final_2897"
# Create a backup folder with a timestamp
backup_dir = f"/root/piper/src/python/feedback_pipline/backup/Narakeet_base_10k_final_2897_backup_{timestamp}"
global_error_csv = "/root/piper/src/python/feedback_pipline/Global_feeback_count/global_error_feedback.csv"
output_csv_dir = f"/root/piper/src/python/feedback_pipline/generated_sentences/"
output_csv_path = f"/root/piper/src/python/feedback_pipline/generated_sentences/Generated_sentences_{timestamp}.csv"
metadata_csv = f"/root/piper/src/python/feedback_pipline/metadata/meta_{timestamp}/metadata.csv"
output_dir = f"/root/piper/src/python/feedback_pipline/Audio_data/base_audio_files_{timestamp}"
# Create the 'readydata' directory and the 'wav' subdirectory
readydata_dir = f"/root/piper/src/python/feedback_pipline/readydata_for_training/process_{timestamp}"
output_dir_preprocess = os.path.expanduser(f'/root/piper/src/python/feedback_pipline/pipeline_training_male/model_{timestamp}')


# Read the feedback CSV
df_feedback = pd.read_csv(feedback_csv_path)

# Filter for rows where the Model is "Celestia X"
df_feedback = df_feedback[df_feedback["Model"].str.strip() == "Celestia X"]

# Check if there is any feedback
if df_feedback.empty:
    print("No feedback found. Exiting pipeline.")
    exit()

# Convert the Timestamp column to datetime objects
df_feedback['Timestamp'] = pd.to_datetime(df_feedback['Timestamp'], format="%Y-%m-%d %H:%M:%S", errors='coerce')
df_feedback = df_feedback[df_feedback['Error Words'].notna()]
# Get the maximum feedback timestamp from the CSV
max_feedback_time = df_feedback['Timestamp'].max()
print("Max feedback timestamp in CSV:", max_feedback_time)

# Load the last processed timestamp from JSON (if exists), else default to the minimum datetime
if os.path.exists(last_processed_json):
    try:
        with open(last_processed_json, 'r') as f:
            data = json.load(f)
            last_processed_str = data.get("last_processed")
            if last_processed_str:
                last_processed_time = datetime.datetime.strptime(last_processed_str, "%Y-%m-%d %H:%M:%S")
            else:
                last_processed_time = datetime.datetime.min
    except Exception as e:
        print("Error loading last processed timestamp from JSON. Defaulting to minimum datetime.", e)
        last_processed_time = datetime.datetime.min
else:
    last_processed_time = datetime.datetime.min

print("Last processed timestamp:", last_processed_time)

# Check if there's new feedback
if max_feedback_time > last_processed_time:
    # Filter the data to only include new feedback since the last run
    df_feedback = df_feedback[df_feedback['Timestamp'] > last_processed_time]
    print("New feedback found. Running pipeline...")
    

    #Backup
    backup_models(src_dir,backup_dir)

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", "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"))


    df_words = df_feedback

    # Filter for rows where the Model is "Celestia X"
    df_words = df_words[df_words["Model"].str.strip() == "Celestia X"]

    # Drop rows where "Error Words" is NaN
    df_words = df_words.dropna(subset=["Error Words"])
    print(df_words)
    # df_words = df_words.tail()  # Adjust this as necessary if you want to use the entire CSV

    # Ensure the CSV contains a "Word" column
    if "Error Words" not in df_words.columns:
        raise ValueError("CSV file must contain a column named 'Word'")


    # Update the global error feedback CSV using the feedback DataFrame
    update_global_error_feedback_from_df(df_words, global_error_csv)

    # Collect all words into a set (for uniqueness)
    unique_words = set()

    # Go through each row in the CSV and split the words
    for idx, row in df_words.iterrows():
        words_str = str(row["Error Words"])
        print(words_str)
        splitted_words = [w.strip() for w in words_str.split(",") if w.strip()]
        for word in splitted_words:
            unique_words.add(word)

    print(unique_words)

    # Parameters
    NUM_SENTENCES_PER_WORD = 50
    WORDS_PER_SENTENCE = 10

    # Generate sentences for each unique word
    sentences_data = []
    sentence_id = 1  # Initialize Sentence ID

    for word in unique_words:
        generated_sentences = generate_sentences_for_word(
            word,
            client,
            num_sentences=NUM_SENTENCES_PER_WORD,
            words_per_sentence=WORDS_PER_SENTENCE
        )
        for sentence in generated_sentences:
            sentences_data.append({
                "Sentence ID": sentence_id,
                "Word": word,
                "Sentence": sentence
            })
            sentence_id += 1

    # Create a DataFrame for the generated sentences
    df_sentences = pd.DataFrame(sentences_data)

    # Save to CSV
    os.makedirs(output_csv_dir, exist_ok=True)
    print(df_sentences)


    df_sentences.to_csv(output_csv_path, index=False, encoding="utf-8")

    print(f"Generated sentences saved to: {output_csv_path}")


    # Narakeet API configuration
    api_key = 'cS1aH0AKx86jC9u4MLbmp9eyhCUuRIlryU3XbgSb'  # Replace with your actual API key
    audio_format = "wav"  # Using audio_format instead of 'format' to avoid naming conflicts
    voice = "imran"

    # Create output directory with a timestamp for uniqueness
    os.makedirs(output_dir, exist_ok=True)

    # Function to show progress
    def show_progress(progress_data):
        print(progress_data)

    # Initialize the Narakeet API (assuming AudioAPI is defined/imported)
    api = AudioAPI(api_key)

    # Read the CSV file containing sentences
    df_input = pd.read_csv(output_csv_path)
    df_input = df_input.dropna(subset=['Sentence']) 
    sentences = df_input['Sentence'].tolist()  # Ensure your CSV column name matches exactly

    # List to collect metadata for each generated audio file
    metadata = []

    # Process each sentence: generate audio and record metadata
    for index, sentence in enumerate(sentences):
        try:
            print(f"Processing sentence {index + 1}: {sentence}")

            # Request an audio generation task
            task = api.request_audio_task(audio_format, sentence, voice)
            task_result = api.poll_until_finished(task['statusUrl'], show_progress)

            # Check if the audio generation succeeded
            if task_result['succeeded']:
                filename = f"sentence_{index + 1}.{audio_format}"
                file_path = os.path.join(output_dir, filename)
                api.download_to_file(task_result['result'], file_path)
                print(f"Downloaded to {file_path}")

                # Append the file path and corresponding sentence to the metadata list
                metadata.append({"path": filename, "sentence": sentence})
            else:
                # Handle failure (you can modify this to continue, log error, etc.)
                raise Exception(f"Error processing sentence {index + 1}: {task_result['message']}")
        except:
            print("error")

    print("All audio files have been processed.")

    # Save the metadata to CSV using '|' as a delimiter, with header row included
    os.makedirs(os.path.dirname(metadata_csv), exist_ok=True)

    df_metadata = pd.DataFrame(metadata)
    # Write the CSV with header (so the first row is "path|sentence")
    df_metadata.to_csv(metadata_csv, sep='|', index=False, header=True, mode='w')

    print(f"Metadata saved to {metadata_csv}")

    os.makedirs(readydata_dir, exist_ok=True)

    readydata_wav_dir = os.path.join(readydata_dir, "wav")
    os.makedirs(readydata_wav_dir, exist_ok=True)

    # Copy all audio files from the output directory to readydata/wav
    for file_name in os.listdir(output_dir):
        if file_name.endswith(f".{audio_format}"):
            source_path = os.path.join(output_dir, file_name)
            dest_path = os.path.join(readydata_wav_dir, file_name)
            shutil.copy(source_path, dest_path)
            print(f"Copied {source_path} to {dest_path}")

    # Copy the metadata CSV to the readydata directory
    dest_metadata_csv = os.path.join(readydata_dir, os.path.basename(metadata_csv))
    shutil.copy(metadata_csv, dest_metadata_csv)
    print(f"Copied {metadata_csv} to {dest_metadata_csv}")

    print("All files have been successfully copied to the readydata folder.")

    # Path to your virtual environment directory
    venv_path = "/root/piper/src/python/.venv"

    # Copy current environment and prepend the venv's bin directory to PATH
    env = os.environ.copy()
    env["PATH"] = os.path.join(venv_path, "bin") + ":" + env["PATH"]

    env["CUDA_VISIBLE_DEVICES"] = "1"


    print("Script is running with the virtual environment!")


    command = [
        "python3", "-m", "piper_train.preprocess",
        "--language", "ur",
        "--input-dir", readydata_dir,
        "--output-dir", output_dir_preprocess,
        "--dataset-format", "ljspeech",
        "--single-speaker",
        "--sample-rate", "16000"
    ]

    result = subprocess.run(command, capture_output=True, text=True, cwd="/root/piper/src/python", env=env)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    if result.returncode != 0:
        print("Preprocessing failed with error:", result.stderr)
    else:
        print("Preprocessing completed successfully.")

    # Use glob to find ckpt files in the folder (non-recursively)
    ckpt_files = glob.glob(os.path.join(src_dir, "*.ckpt"))

    if ckpt_files:
        # If multiple files are found, you can choose the first one or iterate as needed.
        ckpt_file = ckpt_files[0]
        print("Found ckpt file:", ckpt_file)
        
        # Extract the epoch number using a regular expression.
        match = re.search(r"epoch=(\d+)", ckpt_file)
        if match:
            epoch_number = int(match.group(1))
            new_epoch = epoch_number + 2000
            print("Original Epoch Number:", epoch_number)
            print("New Epoch Number (after adding 100):", new_epoch)
        else:
            print("Epoch number not found in the file name.")
    else:
        print("No .ckpt file found in folder:", folder_path)



    train_command = [
        "python3", "-m", "piper_train",
        "--dataset-dir", output_dir_preprocess,
        "--accelerator", "gpu",
        "--devices", "1",
        "--batch-size", "30",
        "--validation-split", "0.0",
        "--num-test-examples", "0",
        "--max_epochs", str(new_epoch),
        "--checkpoint-epochs", "1",
        "--resume_from_checkpoint",f"{ckpt_file}",
        "--precision", "32",
        "--quality", "x-low"
    ]
    # Run the training command and wait for it to complete
    result = subprocess.run(train_command, capture_output=True, text=True, cwd="/root/piper/src/python", env=env)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    print("Training process initiated successfully.")
    # After processing, update the JSON file with the latest timestamp
    
    
    new_data = {"last_processed": max_feedback_time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(last_processed_json, 'w') as f:
        json.dump(new_data, f)
    print("Updated last processed timestamp to:", max_feedback_time)


    source_checkpoints_folder = f"{output_dir_preprocess}/lightning_logs/version_0/checkpoints"
    source_config_file = f"{output_dir_preprocess}/config.json"

    move_checkpoints_and_config(source_checkpoints_folder, source_config_file, src_dir)


else:
    print("No new feedback since the last run. Exiting pipeline.")
    exit()


