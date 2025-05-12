import torch
import streamlit as st
import numpy as np
import os
import csv
import time
import json
import logging
import sys
from pathlib import Path
from enum import Enum
from typing import List
import torch
from piper_train.vits.lightning import VitsModel
from piper_train.vits.utils import audio_float_to_int16
from piper_train.vits.wavfile import write as write_wav
import glob
from piper_phonemize import phonemize_codepoints, phonemize_espeak, tashkeel_run
import re

# Set Streamlit page configuration
st.set_page_config(page_title="Urdu TTS Inference", page_icon="🔊", layout="wide")

# Custom CSS for improved UI aesthetics
st.markdown("""
    <style>
        .stTextArea, .stSelectbox, .stButton button {
            border-radius: 10px;
            font-size: 16px;
            padding: 10px;
        }
        .stButton button {
            background-color: #1E88E5;
            color: white;
            border-radius: 10px;
            padding: 10px;
            font-weight: bold;
            transition: 0.3s;
        }
        .stButton button:hover {
            background-color: #0D47A1;
        }
        .stMarkdown h1 {
            color: #1E88E5;
            font-size: 32px;
            font-weight: bold;
            text-align: center;
        }
        .stMarkdown h2 {
            color: #FFA000;
            font-size: 24px;
        }
        .stSuccess {
            color: green;
            font-weight: bold;
            text-align: center;
            font-size: 18px;
        }
        .stAudio {
            border: 2px solid #4CAF50;
            border-radius: 10px;
            padding: 5px;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

_LOGGER = logging.getLogger("piper_train.infer")

def detect_ckpt_models(models_path):
    ckpt_files = list(Path(models_path).glob("*.ckpt"))
    return [str(f) for f in ckpt_files] if ckpt_files else None

def load_ckpt(model_path):
    print("loading_model!!!!!!!!!!!!!")
    config = load_config(model_path)
    model_ckpt = detect_ckpt_models(model_path)[0]
    model = VitsModel.load_from_checkpoint(str(model_ckpt), dataset=None)
    model.eval()
    with torch.no_grad():
        model.model_g.dec.remove_weight_norm()
    return model, config

def load_config(model_path):
    with open(model_path + '/config.json', "r") as file:
        return json.load(file)
    
PAD = "_"  # padding (0)
BOS = "^"  # beginning of sentence
EOS = "$"  # end of sentence

class PhonemeType(str, Enum):
    ESPEAK = "espeak"
    TEXT = "text"

def phonemize(config, text: str) -> List[List[str]]:
    """Text to phonemes grouped by sentence."""
    if config["phoneme_type"] == PhonemeType.ESPEAK:
        if config["espeak"]["voice"] == "ar":
            # Arabic diacritization
            # https://github.com/mush42/libtashkeel/
            text = tashkeel_run(text)
        return phonemize_espeak(text, config["espeak"]["voice"])
    if config["phoneme_type"] == PhonemeType.TEXT:
        return phonemize_codepoints(text)
    raise ValueError(f"Unexpected phoneme type: {self.config.phoneme_type}")

def phonemes_to_ids(config, phonemes: List[str]) -> List[int]:
    """Phonemes to ids."""
    id_map = config["phoneme_id_map"]
    ids: List[int] = list(id_map[BOS])
    for phoneme in phonemes:
        if phoneme not in id_map:
            print("Missing phoneme from id map: %s", phoneme)
            continue
        ids.extend(id_map[phoneme])
        ids.extend(id_map[PAD])
    ids.extend(id_map[EOS])
    return ids

def inferencing(model, config, text, length_scale=1, noise_scale=0.667, noise_scale_w=0.8, sentence_silence=0.0):
    inference_start_time = time.time()
    audios = []
    text_phonemes = phonemize(config, text)
    num_silence_samples = int(sentence_silence * config["audio"]["sample_rate"])
    silence = np.zeros(num_silence_samples, dtype=np.int16)
    for phonemes in text_phonemes:
        phoneme_ids = phonemes_to_ids(config, phonemes)
        text_tensor = torch.LongTensor(phoneme_ids).unsqueeze(0)
        text_lengths = torch.LongTensor([len(phoneme_ids)])
        scales = [noise_scale, length_scale, noise_scale_w]
        audio = model(text_tensor, text_lengths, scales).detach().numpy()
        audio = audio_float_to_int16(audio.squeeze())
        audio = np.concatenate((audio, silence))
        audios.append(audio)
    merged_audio = np.concatenate(audios)
    sample_rate = config["audio"]["sample_rate"]
    end_time = time.time()
    print(sample_rate)

    audio_duration_sec = merged_audio.shape[-1] / sample_rate
    print("audio duration:", audio_duration_sec)
    infer_sec = end_time - inference_start_time
    print(infer_sec)

    output_path = f"/root/piper/src/python/streamlit_output/audio/{inference_start_time}.wav"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_wav(str(output_path), sample_rate, merged_audio)
    return output_path, sample_rate,audio_duration_sec,infer_sec


# Define model paths.
model_paths = {
    "Sadaa-e-Niswan": "/root/piper/trained_models/female_2500_base_10k",
    "EchoVerse Compact": "/root/piper/trained_models/Quran_denoised_finetuned_10k/",
    "Celestia X": "/root/piper/trained_models/Narakeet_base_10k_final_2897"
}

# Initialize session state for loaded models if not already present.
if "loaded_models" not in st.session_state:
    st.session_state.loaded_models = {}

# Load models on app load.
for model_name, model_path in model_paths.items():
    if model_name not in st.session_state.loaded_models or \
       st.session_state.loaded_models[model_name]["model_path"] != model_path:
        with st.spinner(f"Loading model {model_name}..."):
            print(f"Loading model {model_name}...")
            model, config = load_ckpt(model_path)
            st.session_state.loaded_models[model_name] = {
                "model": model,
                "config": config,
                "model_path": model_path
            }

import re
import csv
import time
import os

# UI Layout
st.title("🔊 Urdu Text-to-Speech Inference App")
st.markdown("Convert Urdu text into natural-sounding speech")

text_input = st.text_area(
    "✍️ Enter Urdu text:",
    "محنت، اور استقامت انسان کی کامیابی کے اہم ستون ہیں۔",
    height=100
)

# Initialize audio results in session state if not present.
if "audio_results" not in st.session_state:
    st.session_state.audio_results = {}
if "error_words" not in st.session_state:
    st.session_state.error_words = {}

# Generate Audio button uses already loaded models.
if st.button("Generate Audio"):
    st.session_state.audio_results = {}  # Reset stored results
    st.session_state.timing_results = {}  # Reset timing results
    for model_name in model_paths.keys():
        with st.spinner(f"Generating with {model_name}..."):
            # Use cached model and configuration.
            model = st.session_state.loaded_models[model_name]["model"]
            config = st.session_state.loaded_models[model_name]["config"]
            audio_path, sample_rate, audio_duration, infer_sec = inferencing(model, config, text_input)
            st.session_state.audio_results[model_name] = audio_path
            st.session_state.timing_results[model_name] = {
                "audio_duration": audio_duration,
                "infer_sec": infer_sec
            }
            # Initialize error words list if not present for this model.
            if model_name not in st.session_state.error_words:
                st.session_state.error_words[model_name] = []

if st.session_state.audio_results:
    st.subheader("Provide feedback on mispronounced words.")
    
    # Remove punctuation from Urdu text before splitting into unique words.
    clean_text = re.sub(r'[^\w\s]', '', text_input, flags=re.UNICODE)
    words = list(dict.fromkeys(clean_text.split()))
    
    with st.form("Mispronunced_feedback_form"):
        cols = st.columns(len(st.session_state.audio_results))
        for i, model_name in enumerate(st.session_state.audio_results.keys()):
            with cols[i]:
                st.markdown(f"### {model_name}")
                st.audio(
                    st.session_state.audio_results[model_name],
                    format="audio/wav",
                    start_time=0
                )
                infer_time = st.session_state.timing_results[model_name]["infer_sec"]
                audio_dur = st.session_state.timing_results[model_name]["audio_duration"]
                st.markdown(f"⏳ **Inference Time:** {infer_time:.2f} sec")
                st.markdown(f"🎵 **Audio Duration:** {audio_dur:.2f} sec")
                selected_errors = st.multiselect(
                    f"❌ Which words (max 5) are pronounced wrong in {model_name}?",
                    options=words,
                    default=st.session_state.error_words[model_name],
                    key=f"error_word_{model_name}"
                )
                st.session_state.error_words[model_name] = selected_errors

        submit_feedback = st.form_submit_button("✅ Submit Feedback")
    
    if submit_feedback:
        # Validate: if any model has more than 5 error words, throw an error.
        error_flag = False
        for model in st.session_state.audio_results:
            if len(st.session_state.error_words[model]) > 5:
                st.error(f"For model {model}, you selected more than 5 error words. Please select at most 5.")
                error_flag = True
        if error_flag:
            st.stop()  # Halt further processing if validation fails.
        
        # Write error feedback to CSV.
        feedback_file_path = "/root/piper/src/python/streamlit_output/model_error_feedback.csv"
        os.makedirs(os.path.dirname(feedback_file_path), exist_ok=True)
        file_exists = os.path.isfile(feedback_file_path) and os.path.getsize(feedback_file_path) > 0
        with open(feedback_file_path, "a", newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow([
                    "Timestamp", "Input Text", "Model",
                    "Error Words", "Inference Time (sec)", "Audio Duration (sec)"
                ])
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            for model in st.session_state.audio_results:
                writer.writerow([
                    timestamp,
                    text_input,
                    model,
                    ", ".join(st.session_state.error_words[model]) if st.session_state.error_words[model] else "None",
                    f"{st.session_state.timing_results[model]['infer_sec']:.2f}",
                    f"{st.session_state.timing_results[model]['audio_duration']:.2f}"
                ])

        # Optionally remove temporary audio files.
        for audio_path in st.session_state.audio_results.values():
            try:
                os.remove(audio_path)
            except FileNotFoundError:
                pass

        # Clear session state related to audio and error words.
        st.session_state.audio_results = {}
        st.session_state.timing_results = {}
        st.session_state.error_words = {}

        st.success("Feedback saved successfully!")
