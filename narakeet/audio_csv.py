import os
import pandas as pd
from narakeet_api import AudioAPI
import datetime

# Narakeet API key
api_key = 'cS1aH0AKx86jC9u4MLbmp9eyhCUuRIlryU3XbgSb'  # Replace with your actual API key
format = "wav"
voice = "Mawra"

duration = 0
total_duration= 0 
target_duration = 10000

# Directory to save audio files
output_dir = F"/root/narakeet/text-to-speech-polling-api-python-example/data/female_1370_{datetime.datetime.now()}"

#output_dir = F"/root/narakeet/urdu_single_sentence.csv"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Function to show progress
def show_progress(progress_data):
    print(progress_data)

# Initialize the Narakeet API
api = AudioAPI(api_key)

# Read the CSV file and extract sentences
df = pd.read_csv('/root/narakeet/Urdu sentences - Sheet1.csv')  # Replace with your CSV file path
print(df.head())
sentences = df['sentences'].tolist()
#sentences = sentences[:1]
print(len(sentences))
# Iterate through each sentence and generate audio
for index, sentence in enumerate(sentences):
    if total_duration>=target_duration:
        break
    #try:
    print(f"Processing sentence {index + 1}: {sentence}")

    # Start a build task using the text sample and voice
    task = api.request_audio_task(format, sentence, voice)
    task_result = api.poll_until_finished(task['statusUrl'], show_progress)
    # Save the audio file
    if task_result['succeeded']:
        headers = task.get('headers', {})

        # Extract duration if available
        duration = float(task_result.get('durationInSeconds', 0))
        total_duration+=duration

        print(f"Audio Duration: {duration} seconds")
        print(f"Total Duration: {total_duration} seconds")


        filename = os.path.join(output_dir, f'sentence_{index +1}.{format}')
        api.download_to_file(task_result['result'], filename)
        print(f"Downloaded to {filename}")
    else:
        raise Exception(task_result['message'])
    #except:
    #    print("error in script")

print("All audio files have been downloaded.")

# import os
# import shutil

# # Define the source directory
# source_dir = "/root/narakeet/text-to-speech-polling-api-python-example/data"
# # Define the destination directory
# dest_dir = os.path.join(source_dir, "merged_files_2")

# # Create the destination folder if it doesn't exist
# os.makedirs(dest_dir, exist_ok=True)

# # Walk through all subdirectories and copy files
# for root, _, files in os.walk(source_dir):
#     if root == dest_dir:  # Avoid copying the merged_files folder itself
#         continue
#     for file in files:
#         source_path = os.path.join(root, file)
#         dest_path = os.path.join(dest_dir, file)

#         # If a file with the same name exists, append a number to avoid conflicts
#         counter = 1
#         while os.path.exists(dest_path):
#             filename, ext = os.path.splitext(file)
#             dest_path = os.path.join(dest_dir, f"{filename}_{counter}{ext}")
#             counter += 1

#         shutil.copy2(source_path, dest_path)
#         print(f"Copied: {source_path} -> {dest_path}")

# print("All files copied successfully!")

# Processing sentence 501: ایک سیاسی جماعت کی بےجا حمایت پر ان کے جج بننے کے بھی خلاف تھا۔
# Processing sentence 500: کینسر کے شکار رومن رینز کی حالت اب کیسی ہے

