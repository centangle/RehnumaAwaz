### To setup the Piper repo:

Ignore any warnings you get in the following steps:

python version used : 3.11.2

```

git clone https://github.com/rhasspy/piper.git

apt install python3.11-venv
python3 -m venv ~/piper/src/python/.venv

cd ~/piper/src/python/
source ~/piper/src/python/.venv/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install --upgrade wheel setuptools
python3 -m pip install -e .
python3 -m pip install torchmetrics==0.11.4

apt update &&  apt install -y build-essential
pip install --upgrade pip setuptools wheel
pip install cython numpy
pip install pytorch-lightning librosa

bash build_monotonic_align.sh

ignore all futures warnings you get while following these steps


```
### ALTERNATIVE: Use UrduTTS Docker image
1. Untar the docker image
```
docker load -i /path/to/urdu_tts.tar

```
2. Run the Container
```
docker run -dit --gpus all --name urdu_tts_container \
    --shm-size=8g \
    -v /root/UrduTTS/data:/app/data \
    urdu_tts /bin/bash
```

Now to run any script in the docker, enter into the docker using the following command: 
```
docker exec -it urdu_tts_container /bin/bash

```

### To Preprocess the dataset:
1- set input directory path..
  * Input directory contains "wav" directory, with audio files, and a metadata.csv file.
  * metadata.csv files should have the following structure:
    * path|sentence

  or in case of multispeaker:
    * path|speaker|sentence

  where path is just the name of the file, and speaker is ID of the speaker.
  
2- You can define any output directory, where config.json, dataset.json will be prepared which contains list of phonemes generated in the preprocessing.

3- For Low quality model, set the sample-rate to 16000, and for medium or high quality model, set the sample-rate to 22050

4- If the dataset is single speaker, set single-speaker argument.

Note: For the dockerized solution, create a "data" directory, along with two sub-directories:
  * raw_data: contains the dataset in the above format, wav files and metadata.csv
  * models: contains the preprocessed dataset, which to be used in training stage. The output of preprocessing will be saved here.  


Refer to the code below to preprocess the dataset:


```

python3 -m piper_train.preprocess \
  --language en \
  --input-dir ~/piper/my-dataset \
  --output-dir ~/piper/my-training \
  --dataset-format ljspeech \
  --single-speaker \
  --sample-rate 22050
```



### To Train the model:

Once the dataset has been processed, you can set the **dataset-dir** argument with the path of the preprocessed dataset directory (which contains config, dataset.json files).

For resuming from checkpoint use the argument **resume_from_checkpoint**.

You can set the quality using the argument **quality**
```
--quality high
--quality medium
--quality x-low

```
Use the following command to start training:

```
cd ~/piper/src/python/

python3 -m piper_train \
    --dataset-dir ~/piper/my-training \
    --accelerator 'gpu' \
    --devices 1 \
    --batch-size 32 \
    --validation-split 0.0 \
    --num-test-examples 0 \
    --max_epochs 6000 \
    --resume_from_checkpoint ~/piper/epoch=2164-step=1355540.ckpt \
    --checkpoint-epochs 1 \
    --precision 32

```

By default, the model checkpoints will get saved in the preprocessed dataset directory.


### To add Narakeet:

In **root/naraket** directory, the repo is cloned. You can use the **audio_csv.py** file, to get the text from the csv file and pass it to narakeet api, which will return the audio. You can set the input csv and output paths.

Once the audio files are downloaded, you can create the metadata.csv file in the same manner as required in the preprocess step (path|sentence). Make sure the correct file name is assigned to the sentence.

Once the data has been prepared. You can run the preprocess step, and train the model.




# Audio Generation and Model Training Pipelines

This project provides two automated pipelines designed to process mispronounced feedback data, generate new training sentences and audio files, and subsequently retrain models. There are two distinct pipelines:

- **Male Pipeline:** Processes feedback for the "Celestia X" model.
- **Female Pipeline:** Processes feedback for the "Sadaa-e-Niswan" model.

Both pipelines follow a similar workflow but operate on different subsets of feedback and use different configuration parameters (e.g., different voices for audio generation).

---

## Project Structure

- **last_processed_run_male.json**  
  Stores the timestamp of the last processed feedback for the male pipeline.

- **last_processed_run_female.json**  
  Stores the timestamp of the last processed feedback for the female pipeline.

- **run_daily.sh**  
  A shell script that activates the project’s virtual environment and runs both pipelines sequentially. It logs output with timestamped log files for each pipeline run.

- **utils.py**  
  Contains helper functions used by both pipelines, including:
  - **backup_models:** Backs up the current model directory before retraining.
  - **clean_urdu_text:** Cleans and normalizes Urdu text.
  - **generate_sentences_for_word:** Uses the OpenAI API to generate Urdu sentences containing error words.
  - **update_global_error_feedback_from_df:** Updates a global CSV tracking error words, counts, and timestamps.
  - **move_checkpoints_and_config:** Manages moving checkpoint files and configuration after training.

- **pipeline.py**  
  The male pipeline script. It:
  - Reads error feedback from a CSV (`/root/piper/src/python/streamlit_output/model_error_feedback.csv`) and filters for the "Celestia X" model.
  - Compares feedback timestamps with the last processed time stored in `last_processed_run_male.json`.
  - If new feedback exists, it backs up the current model, generates new sentences for each unique error word, and saves them to a CSV.
  - Uses the Narakeet API to convert sentences to audio files.
  - Prepares training data and retrains the model using the `piper_train` module.
  - Updates the last processed timestamp and moves checkpoint/config files post-training.

- **pipeline_female.py**  
  The female pipeline script. Its workflow is similar to `pipeline.py`, but:
  - Filters for the "Sadaa-e-Niswan" model.
  - Uses different backup directories and configuration (e.g., voice parameter set to "mawra").
  - Saves generated sentences, audio files, metadata, and training data in female-specific directories.

---

## Directory and Path Configuration

This project uses hard-coded paths for various resources and outputs. It is essential to set these up properly to match your environment and directory structure. Below are key paths you may need to update:

- **Feedback CSV:**  
  - Path: `/root/piper/src/python/streamlit_output/model_error_feedback.csv`  
    Ensure that the CSV file exists at this location and includes columns such as "Model", "Error Words", and "Timestamp".

- **Last Processed Timestamps:**  
  - Male: `/root/piper/src/python/feedback_pipline/last_processed_run_male.json`  
  - Female: `/root/piper/src/python/feedback_pipline/last_processed_run_female.json`  
    These files store the last feedback processing timestamps and must be writable.

- **Backup Directories:**  
  - For male pipeline: The backup is created from `/root/piper/trained_models/Narakeet_base_10k_final_2897`.
  - For female pipeline: The backup is created from `/root/piper/trained_models/female_2500_base_10k`.
  - Backup folders are created in the corresponding `backup` subdirectory inside `/root/piper/src/python/feedback_pipline/`.

- **Generated Sentences and Audio Files:**  
  - Male generated sentences are saved in `/root/piper/src/python/feedback_pipline/generated_sentences/`.  
  - Female generated sentences are saved in `/root/piper/src/python/feedback_pipline/generated_sentences_female/`.
  - Male audio outputs are saved in `/root/piper/src/python/feedback_pipline/Audio_data/`.
  - Female audio outputs are saved in `/root/piper/src/python/feedback_pipline/Audio_data_female/`.

- **Metadata and Ready Data for Training:**  
  - Male metadata CSVs and training-ready data are stored under `/root/piper/src/python/feedback_pipline/metadata/` and `/root/piper/src/python/feedback_pipline/readydata_for_training/`, respectively.
  - Female metadata CSVs and training-ready data are stored under `/root/piper/src/python/feedback_pipline/metadata_female/` and `/root/piper/src/python/feedback_pipline/readydata_for_training_female/`, respectively.

- **Virtual Environment:**  
  - The virtual environment is expected at `/root/piper/src/python/.venv/`. The `run_daily.sh` script sources this environment before executing the pipelines.

- **Training Module:**  
  - The retraining steps invoke the `piper_train` module from within `/root/piper/src/python/`. Ensure that the training module and its dependencies are correctly installed and that paths match your project setup.

*Note:* Adjust these paths if your project directory is located elsewhere or if you need a custom configuration for different environments.

---

## Workflow Overview

1. **Feedback Processing:**  
   Each pipeline reads error feedback from the CSV file, filtering based on model name and only processing records with new timestamps.

2. **Sentence Generation:**  
   Unique error words are extracted and, for each word, the OpenAI API generates multiple creative Urdu sentences using the `generate_sentences_for_word` function.

3. **Audio Generation:**  
   Generated sentences are converted into audio files using the Narakeet API. Metadata (file paths and corresponding sentences) is recorded in a CSV file.

4. **Training Data Preparation & Model Retraining:**  
   Audio files and metadata are organized into a "readydata" folder formatted as expected by the training module. The `piper_train` module is then invoked to retrain the model.

5. **Backup and Cleanup:**  
   Before retraining, the current model directory is backed up. After training, checkpoint and configuration files are moved to ensure that the latest outputs are preserved.

6. **Daily Execution:**  
   The `run_daily.sh` script automates the execution of both pipelines daily, logging outputs with timestamps.

---

## Setup and Usage

- **Environment Setup:**  
  Ensure that the Python virtual environment is set up and activated. The virtual environment is located at `/root/piper/src/python/.venv/`.

- **API Keys:**  
  Set up your API keys for the OpenAI API and Narakeet API. The keys can be provided as environment variables or are hard-coded as defaults in the pipeline scripts.

- **Directory Structure:**  
  Verify that the paths for logs, backups, generated sentences, audio files, metadata, and training outputs exist or can be created. Adjust the paths if your project directory differs.

- **Running the Pipelines:**  
  To execute both pipelines, run:
  ```bash
  bash run_daily.sh



# Urdu TTS Inference Streamlit App

This Streamlit application converts Urdu text into natural-sounding speech using multiple pretrained models. It performs inference to generate audio outputs from text and then allows users to provide feedback on mispronunciations. The feedback is logged into a CSV file for further processing or model improvement.

---

## Features

- **Multi-Model Inference:**  
  The app loads three distinct models:
  - **Sadaa-e-Niswan:** Located at `/root/piper/trained_models/female_2500_base_10k`
  - **EchoVerse Compact:** Located at `/root/piper/trained_models/Quran_denoised_finetuned_10k/`
  - **Celestia X:** Located at `/root/piper/trained_models/Narakeet_base_10k_final_2897`

- **Text-to-Speech (TTS) Conversion:**  
  Converts the input Urdu text into speech by transforming text into phonemes and then performing inference with the loaded models.

- **Customizable UI:**  
  Uses custom CSS to enhance the user interface and improve usability with styled buttons, text areas, and audio players.

- **Feedback Collection:**  
  After audio generation, users can select and submit a list of mispronounced words (up to five per model). This feedback is saved into a CSV file.

- **Inference Metrics:**  
  Displays inference time and audio duration to help gauge the performance of each model.

---

## Directory and Path Configuration

The application relies on several hard-coded paths that you may need to adjust to match your environment:

- **Model Directories:**  
  - **Sadaa-e-Niswan:** `/root/piper/trained_models/female_2500_base_10k`  
  - **EchoVerse Compact:** `/root/piper/trained_models/Quran_denoised_finetuned_10k/`  
  - **Celestia X:** `/root/piper/trained_models/Narakeet_base_10k_final_2897`

- **Output Audio Files:**  
  Generated WAV files are stored in:  
  `/root/piper/src/python/streamlit_output/audio/`

- **Feedback CSV:**  
  The error feedback from users is recorded in:  
  `/root/piper/src/python/streamlit_output/model_error_feedback.csv`

- **Streamlit App Code:**  
  The main app file is `inference_streamlit.py`.

*Note:* If your project directory or deployment environment differs from the above structure, be sure to update the corresponding paths within the code.

---

## Setup and Installation

1. **Clone the Repository:**  
   Clone or download the repository containing this Streamlit app.

2. **Install Dependencies:**  
   Ensure you have Python installed (preferably Python 3.7+). Install the required packages using pip:
   ```bash
   pip install streamlit torch numpy pandas piper_train piper_phonemize
  
## Running the App

1. **Navigate to the App Directory::** 
   Open a terminal and change to the directory containing inference_streamlit.py.

2. **Launch the Streamlit App:** 
   Run the following command:

  `streamlit run inference_streamlit.py`

2. **Interacting with the App:** 

   Enter the Urdu text you wish to convert in the provided text area.

   Click Generate Audio to produce audio outputs from all available models.

   Listen to the generated audio files and provide feedback on any mispronounced words using the checkboxes.

   Submit your feedback, which then gets recorded into a CSV file at `/root/piper/src/python/streamlit_output/model_error_feedback.csv`.
