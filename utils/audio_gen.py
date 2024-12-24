from pydub import AudioSegment
import datetime
import os
import glob
import re
import boto3
from contextlib import closing


def generate_host(text: str, polly_client, output_dir: str):
    now = int(datetime.datetime.now().timestamp())
    response = polly_client.synthesize_speech(
        Engine='neural',
        Text=text,
        OutputFormat='mp3',
        VoiceId='Matthew'
    )

    output_path = f"./{output_dir}/host_{now}.mp3"
    with closing(response['AudioStream']) as stream:
        with open(output_path, 'wb') as file:
            file.write(stream.read())
    return output_path


def generate_expert(text: str, polly_client, output_dir: str):
    now = int(datetime.datetime.now().timestamp())
    response = polly_client.synthesize_speech(
        Engine='neural',
        Text=text,
        OutputFormat='mp3',
        VoiceId='Stephen'
    )

    output_path = f"./{output_dir}/expert_{now}.mp3"
    with closing(response['AudioStream']) as stream:
        with open(output_path, 'wb') as file:
            file.write(stream.read())
    return output_path


def generate_learner(text: str, polly_client, output_dir: str):
    now = int(datetime.datetime.now().timestamp())
    response = polly_client.synthesize_speech(
        Engine='neural',
        Text=text,
        OutputFormat='mp3',
        VoiceId='Joanna'
    )
    output_path = f"./{output_dir}/learner_{now}.mp3"
    with closing(response['AudioStream']) as stream:
        with open(output_path, 'wb') as file:
            file.write(stream.read())
    return output_path


def merge_mp3_files(directory_path, output_file):
    # Find all .mp3 files in the specified directory
    mp3_files = [os.path.basename(x)
                 for x in glob.glob(f"./{directory_path}/*.mp3")]

    # Sort files by datetime extracted from filename
    sorted_files = sorted(
        mp3_files,
        key=lambda x: re.search(r"(\d{10})", x).group(0)
    )
    # Initialize an empty AudioSegment for merging
    merged_audio = AudioSegment.empty()

    # Merge each mp3 file in sorted order
    for file in sorted_files:
        audio = AudioSegment.from_mp3(f"./{directory_path}/{file}")
        merged_audio += audio

    # Export the final merged audio
    merged_audio.export(output_file, format="mp3")
    print(f"Merged file saved as {output_file}")


def generate_podcast(script, polly_client):
    # create a new directory to store the audio files
    output_dir = f"podcast_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    os.mkdir(output_dir)

    lines = re.findall(
        r"(Host|Learner|Expert):\s*(.*?)(?=(Host|Learner|Expert|$))", script, re.DOTALL
    )

    for speaker, text, _ in lines:
        text = text.strip()

        if speaker == "Host":
            generate_host(text, polly_client, output_dir)
        elif speaker == "Learner":
            generate_learner(text, polly_client, output_dir)
        elif speaker == "Expert":
            generate_expert(text, polly_client, output_dir)

    now = int(datetime.datetime.now().timestamp())
    merge_mp3_files(output_dir, f"podcast_{now}.mp3")
