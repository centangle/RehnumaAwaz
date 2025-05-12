import os
from narakeet_api import AudioAPI

api_key = os.environ['NARAKEET_API_KEY']
format = "wav"
voice = "victoria"
vtt_file = "example.vtt"
voice_speed = 1.1

def show_progress(progress_data):
    # change this to do something smarter, you can use percent and message
    print(progress_data)

api = AudioAPI(api_key)

# start a build task using the VTT file and voice
task = api.request_vtt_task(vtt_file, voice, voice_speed, format)
# and wait for it to finish
task_result = api.poll_until_finished(task['statusUrl'], show_progress)

# grab the result file
if task_result['succeeded']:
    filename = f'output.{format}'
    api.download_to_file(task_result['result'], filename)
    print(f'downloaded to {filename}')
else:
    raise Exception(task_result['message'])
