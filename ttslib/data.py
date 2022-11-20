import json
import os
import re
import time

from lxml.etree import HTML
import moviepy.editor as editor
import requests
from pydub import AudioSegment


def download_subtitle(url, save_file):
    for _ in range(3):
        try:
            data = requests.get(url).json()
            with open(save_file, "w", encoding="utf-8") as fo:
                fo.write(json.dumps(data["body"], ensure_ascii=False))
            return True
        except Exception:
            print("download subtitle error")
            time.sleep(0.5)
    return False


def extract_audio(video_file, save_file):
    video = editor.VideoFileClip(video_file)
    video.audio.write_audiofile(save_file)
    video.close()


def clip_audio(source_file, save_file, start, end):
    audio = AudioSegment.from_wav(source_file)
    audio = audio[start:end]
    audio.export(save_file, format="wav")


def get_index(target_dir, pattern):
    p = re.compile(pattern)
    files = [file for file in os.listdir(target_dir) if p.match(file)]
    if files:
        last_file = max(files, key=lambda x: int(p.match(x).group(1)))
        idx = int(p.match(last_file).group(1)) + 1
    else:
        idx = 0
    return idx


def segment_audio(audio_file, subtitle_file, save_dir, transcript_file, speaker, offset=300):
    os.makedirs(save_dir, exist_ok=True)

    with open(subtitle_file, encoding="utf-8") as fi:
        subtitle = json.load(fi)
    idx = get_index(save_dir, "segment(\d+).wav")
    with open(transcript_file, "a", encoding="utf-8") as fo:
        p = re.compile("[a-zA-Z1-9]+")
        for part in subtitle:
            if len(part["content"]) < 4 or p.search(part["content"]) is not None:
                continue
            save_file = f"segment{idx}.wav"
            start = part["from"] * 1000
            end = part["to"] * 1000 + offset
            if end - start < 1500:
                continue
            clip_audio(audio_file, f"{save_dir}/{save_file}", start, end)
            fo.write(f"{save_dir}/{save_file}\t{part['content']}\t{speaker}\n")
            idx += 1


def get_json_candidates(text):
    num = 0
    start = -1
    for i, chr in enumerate(text):
        if chr == "{":
            if num == 0:
                start = i
            num += 1
        elif chr == "}":
            num -= 1
            if num == 0:
                yield text[start:i+1]


def get_subtitle_url(page_url):
    text = requests.get(page_url).text
    html = HTML(text)
    cur_page = int(page_url[page_url.rfind("p=")+2:])
    print(cur_page)
    for script in html.xpath("//script/text()"):
        if script.startswith("window.__INITIAL_STATE__="):
            for candidate in get_json_candidates(script):
                if all(kw in candidate for kw in ["aid", "cid", "bvid", "title"]):
                    json_data = json.loads(candidate)
                    aid = json_data["videoData"]["aid"]
                    bvid = json_data["videoData"]["bvid"]
                    for page in json_data["videoData"]["pages"]:
                        if page["page"] == cur_page:
                            cid = page["cid"]
                            break
                    url = f"https://api.bilibili.com/x/player/v2?cid={cid}&aid={aid}&bvid={bvid}"
                    data = requests.get(url).json()["data"]
                    subtitle = data.get("subtitle")
                    if subtitle:
                        items = subtitle.get("subtitles")
                        if items:
                            return "https:" + items[0]["subtitle_url"]


def process_video(video_url, save_dir, speaker, sr=44000, offset=300):
    os.makedirs(save_dir, exist_ok=True)
    subtitle_url = get_subtitle_url(video_url)
    print(subtitle_url)
    if subtitle_url is None:
        print("subtitle url not found")
        return
    cache_dir = "data/cache"
    os.makedirs(cache_dir, exist_ok=True)
    idx = get_index(cache_dir, f"subtitle_{speaker}_(\d+).json")
    subtitle_file = f"{cache_dir}/subtitle_{speaker}_{idx}.json"
    idx = get_index(cache_dir, f"audio_{speaker}_(\d+).wav")
    audio_file = f"{cache_dir}/audio_{speaker}_{idx}.wav"

    if not download_subtitle(subtitle_url, subtitle_file):
        print("fail to download subtitle")
        os.system(f"del {subtitle_file}")
        return
    try:
        os.system(f"cd {cache_dir} && you-get --format=dash-flv360 {video_url}")
        video_files = filter(lambda x: any(x.endswith(postfix) for postfix in [".flv", "mp4"]), os.listdir(cache_dir))
        xml_files = [f"{cache_dir}/{file}" for file in os.listdir(cache_dir) if file.endswith(".xml")]
        for file in xml_files:
            os.remove(file)

        video_file = list(video_files)[0]
        new_video_file = f"{cache_dir}/video{video_file[-4:]}"
        os.rename(f'{cache_dir}/{video_file}', new_video_file)

        extract_audio(new_video_file, audio_file)
        os.system(f"ffmpeg -i {audio_file} -ac 1 -ar {sr} temp.wav")
        os.remove(audio_file)
        os.system(f"move temp.wav {audio_file}")

        print("start segmenting")
        segment_audio(audio_file, subtitle_file, save_dir, f"{save_dir}/transcript.txt", speaker, offset)
        os.remove(new_video_file)
    except Exception:
        os.system(f"del {subtitle_file}")
        os.system(f"del {audio_file}")


def collect_transcript():
    with open("data/metadata.txt", "w", encoding="utf-8") as fo:
        for name in os.listdir("data"):
            dir_name = f"data/{name}"
            if os.path.isdir(dir_name) and os.path.exists(f"{dir_name}/transcript.txt"):
                with open(f"{dir_name}/transcript.txt", encoding="utf-8") as fi:
                    for line in fi:
                        splits = line.strip().split("\t")
                        assert len(splits) == 2
                        splits.append(name)
                        fo.write("\t".join(splits) + "\n")


if __name__ == "__main__":

    process_video("https://www.bilibili.com/video/BV1bP4y127TK?p=3", "data/wav/meizi1", "meizi1")

    # collect_transcript()
