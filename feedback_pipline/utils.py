import openai
import pandas as pd
import os
import re
import glob
from narakeet_api import AudioAPI
import datetime
import shutil


def backup_models(src_dir,backup_dir):

    # Ensure the backup parent directory exists
    os.makedirs(os.path.dirname(backup_dir), exist_ok=True)

    # Copy the entire folder to the backup location
    shutil.copytree(src_dir, backup_dir)

    print(f"Backup of {src_dir} has been taken to {backup_dir}")

def clean_urdu_text(text):
    text = re.sub(r"^\d+[\.\-\)]?\s*", "", text) 
    text = re.sub(r"[^ا-یء،۔ ]+", "", text) 
    return text.strip()

# Function to generate multiple sentences for a given word
def generate_sentences_for_word(word,client, num_sentences=1, words_per_sentence=2):

    prompt = f"""
    براہ کرم صرف {num_sentences} منفرد اور تخلیقی اردو جملے تیار کریں جن میں لفظ '{word}' شامل ہو۔
    ہر جملے میں صرف تقریباً {words_per_sentence} الفاظ شامل ہوں؛ براہ کرم ہر جملے میں اضافی الفاظ شامل نہ کریں۔
    جملے صرف اردو زبان میں ہوں، اور کسی بھی جملے میں انگریزی، نمبر، یا کوئی اضافی نشانات نہ ہوں۔
    جملے الگ الگ ہوں اور ہر جملے کے شروع میں کوئی نمبر یا الفاظ شامل نہ ہوں۔
    صرف اتنے جملے فراہم کریں جتنے کہ {num_sentences} ہیں، کوئی اضافی جملہ نہ دیں۔
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "آپ ایک اردو زبان کے ماہر ہیں جو الفاظ پر مبنی جملے تخلیق کر سکتے ہیں۔"},
            {"role": "user", "content": prompt}
        ]
    )

    generated_text = response.choices[0].message.content.strip()
    print(generated_text)
    sentences = generated_text.split("\n")

    # Clean each sentence to ensure only Urdu text (removes numbering, English, extra characters)
    cleaned_sentences = [clean_urdu_text(sentence) for sentence in sentences]
    print(cleaned_sentences)
    return cleaned_sentences

def update_global_error_feedback_from_df(feedback_df, global_csv_path):
    """
    Update a global CSV with error words, their count, and the last feedback time.
    This function checks if 'Feedback Time' is present in the provided DataFrame.
    For each row, it splits the error words and uses the row's feedback time (or current time if missing)
    to update the global CSV.
    """
    # Try to load the existing global CSV; if not present, create a new DataFrame
    try:
        df_global = pd.read_csv(global_csv_path)
        # Ensure proper types for count and timestamp
        df_global["Count"] = pd.to_numeric(df_global["Count"], errors='coerce').fillna(0).astype(int)
    except FileNotFoundError:
        df_global = pd.DataFrame(columns=["Error Word", "Count", "Last Feedback", "Feedback History", "Script Run History"])

    # Compute the current script run time (this timestamp will be recorded for every update)
    script_run_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Check if the feedback DataFrame has a "Feedback Time" column.
    has_feedback_time = "Timestamp" in feedback_df.columns

    # Process each row in the feedback DataFrame
    for idx, row in feedback_df.iterrows():
        # Get the feedback time from the row if available; otherwise use current time
        if has_feedback_time:
            feedback_time_str = row["Timestamp"]
            try:
                # Adjust the datetime format if necessary
                feedback_time = datetime.datetime.strptime(feedback_time_str, "%Y-%m-%d %H:%M:%S")
            except Exception as e:
                print(f"Error parsing feedback time for row {idx}: {e}. Using current time.")
                feedback_time = datetime.datetime.now()
        else:
            feedback_time = datetime.datetime.now()

        feedback_time_formatted = feedback_time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Split error words from the "Error Words" column (assumes comma-separated)
        words_str = str(row["Error Words"])
        splitted_words = [w.strip() for w in words_str.split(",") if w.strip()]
        
        for word in splitted_words:
            if word in df_global["Error Word"].values:
                # Increment the count.
                df_global.loc[df_global["Error Word"] == word, "Count"] += 1

                # Update Feedback History: ensure no duplicate timestamp is added.
                current_history = df_global.loc[df_global["Error Word"] == word, "Feedback History"].iloc[0]
                if pd.isna(current_history) or current_history == "":
                    new_history = feedback_time_formatted
                else:
                    # Split existing timestamps and add only if not present.
                    existing_feedbacks = [ts.strip() for ts in current_history.split(";")]
                    if feedback_time_formatted not in existing_feedbacks:
                        new_history = current_history + ";" + feedback_time_formatted
                    else:
                        new_history = current_history
                df_global.loc[df_global["Error Word"] == word, "Feedback History"] = new_history

                # Update the last feedback time if the new time is later.
                current_time_str = df_global.loc[df_global["Error Word"] == word, "Last Feedback"].iloc[0]
                try:
                    current_time = datetime.datetime.strptime(current_time_str, "%Y-%m-%d %H:%M:%S")
                except Exception as e:
                    current_time = datetime.datetime.min
                if feedback_time > current_time:
                    df_global.loc[df_global["Error Word"] == word, "Last Feedback"] = feedback_time_formatted

                # Update Script Run History only if the current run time is not already recorded.
                current_script_history = df_global.loc[df_global["Error Word"] == word, "Script Run History"].iloc[0]
                if pd.isna(current_script_history) or current_script_history == "":
                    new_script_history = script_run_time
                else:
                    # Split the existing history by comma and check if the current timestamp exists.
                    existing_runs = [ts.strip() for ts in current_script_history.split(",")]
                    if script_run_time not in existing_runs:
                        new_script_history = current_script_history + "," + script_run_time
                    else:
                        new_script_history = current_script_history
                df_global.loc[df_global["Error Word"] == word, "Script Run History"] = new_script_history

            else:
                # For a new error word, initialize its count, last feedback time, feedback history, and script run history.
                new_row = {
                    "Error Word": word,
                    "Count": 1,
                    "Last Feedback": feedback_time_formatted,
                    "Feedback History": feedback_time_formatted,
                    "Script Run History": script_run_time
                }
                df_global = pd.concat([df_global, pd.DataFrame([new_row])], ignore_index=True)
    
    # Ensure that the directory exists before saving the CSV.
    os.makedirs(os.path.dirname(global_csv_path), exist_ok=True)
    df_global.to_csv(global_csv_path, index=False)
    print(f"Global error feedback updated and saved to {global_csv_path}")

def move_checkpoints_and_config(source_checkpoints_folder, source_config_file, destination_folder):

    # Clear the destination folder first
    if os.path.exists(destination_folder):
        for file in os.listdir(destination_folder):
            dest_file_path = os.path.join(destination_folder, file)
            if os.path.isfile(dest_file_path):
                print(f"Removing {dest_file_path}")
                os.remove(dest_file_path)
    else:
        os.makedirs(destination_folder, exist_ok=True)

    # Move all files from the checkpoints folder
    for file in os.listdir(source_checkpoints_folder):
        src_path = os.path.join(source_checkpoints_folder, file)
        dest_path = os.path.join(destination_folder, file)
        if os.path.isfile(src_path):
            print(f"Copying {src_path} -> {dest_path}")
            shutil.copy2(src_path, dest_path)

    # Move the config.json file
    if os.path.isfile(source_config_file):
        dest_config_path = os.path.join(destination_folder, os.path.basename(source_config_file))
        print(f"Copying {source_config_file} -> {dest_config_path}")
        shutil.copy2(source_config_file, dest_config_path)
    else:
        print(f"[Warning] config.json not found at: {source_config_file}")

    print("\nAll files copied to destination.")