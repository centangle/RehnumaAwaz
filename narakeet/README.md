# Narakeet Long content (polling) text to speech API example in Python

This repository provides a quick example demonstrating how to access the
Narakeet [Long Content API](https://www.narakeet.com/docs/automating/text-to-speech-api/#polling) from Python.

The long content API is suitable for large audio conversion tasks, and can
produce professional quality uncompressed WAV files using realistic text to
speech.

Note that Narakeet also has a simpler text to speech API, suitable for smaller conversion tasks, that directly streams back the results. 
See the [Streaming Text to Speech API Example](https://github.com/narakeet/text-to-speech-api-python-example) for more information on how to use that.

The example sends a request to generate an audio file, then downloads the resulting audio into a local file. 

The example uses the [requests](https://requests.readthedocs.io/en/latest/) python library to send HTTPS requests to the Narakeet API.

## Prerequisites

To use this example, you will need Python (3.7 or more recent), and an API key for Narakeet.

## Running the example

1. set and export a local environment variable called `NARAKEET_API_KEY`, containing your API key (or modify [audio.py](audio.py) line 4 to include your API key).
2. optionally edit [audio.py](audio.py) and modify the output file type, voice, script and the function that handles progress notification (lines 5, 6, 7, and 9).
3. run `pip install -r requirements.txt` to install the required libraries
4. run `python audio.py` to create the output audio.

## Converting VTT subtitle files

1. set and export a local environment variable called `NARAKEET_API_KEY`, containing your API key (or modify [convert-vtt.py](convert-vtt.py) line 4 to include your API key).
2. optionally edit [convert-vtt.py](convert-vtt.py) and modify the input file, format, voice and speed (lines 5-8).
3. run `pip install -r requirements.txt` to install the required libraries
4. run `python convert-vtt.py` to create the output audio.

## More information

Check out <https://www.narakeet.com/docs/automating/text-to-speech-api/> for more information on the Narakeet API features.


