import glob
import json
import os
import re

import moviepy.editor as editor
import requests
from pydub import AudioSegment
from vad import segment_audio
import librosa


def extract_audio(video_file, save_file):
    video = editor.VideoFileClip(video_file)
    video.audio.write_audiofile(save_file)
    video.close()


def clip_audio(source_file, save_file, start, end):
    audio = AudioSegment.from_wav(source_file)
    audio = audio[start:end]
    audio.export(save_file, format="wav")


def segment_audio_by_vad(audio_file, save_dir, min_duration=1, max_duration=5):
    os.makedirs(save_dir, exist_ok=True)
    files = glob.glob1(save_dir, "segment*")
    if files:
        p = re.compile("segment(\d+).wav")
        last_file = max(files, key=lambda x: int(p.match(x).group(1)))
        idx = int(p.match(last_file).group(1)) + 1
    else:
        idx = 0
    segment_audio(audio_file, save_dir)
    pattern = re.compile("chunk(\d+).wav")
    for file in sorted(glob.glob1(save_dir, "chunk*"), key=lambda x: int(pattern.match(x).group(1))):
        file = f"{save_dir}/{file}"
        if min_duration < librosa.get_duration(filename=file) < max_duration:
            os.rename(file, f"{save_dir}/segment{idx}.wav")
            idx += 1
        else:
            os.remove(file)


def process_video(video_url, save_dir, min_duration=2, max_duration=8):
    os.makedirs(save_dir, exist_ok=True)
    os.system(f"cd {save_dir} && you-get {video_url}")

    video_files = filter(lambda x: any(x.endswith(postfix) for postfix in [".flv", "mp4"]), os.listdir(save_dir))
    xml_files = [f"{save_dir}/{file}" for file in os.listdir(save_dir) if file.endswith(".xml")]
    for file in xml_files:
        os.remove(file)
    video_file = list(video_files)[0]
    new_video_file = f"{save_dir}/video{video_file[-4:]}"
    os.rename(f'{save_dir}/{video_file}', new_video_file)
    audio_file = f"{save_dir}/audio.wav"
    extract_audio(new_video_file, audio_file)
    os.system(f"ffmpeg -i {audio_file} -acodec pcm_s16le -ac 1 -ar 16000 temp.wav")
    os.remove(audio_file)
    os.system(f"move temp.wav {audio_file}")
    print("start segmenting")
    segment_audio_by_vad(audio_file, f"{save_dir}/segments", min_duration, max_duration)
    os.remove(new_video_file)


if __name__ == "__main__":
   process_video("https://www.bilibili.com/video/BV1S64y117td?p=5", "data/wav/meizi2")
