import argparse
from pathlib import Path
from statistics import mean
import tempfile
import threading
import time

import cv2
from yt_dlp import YoutubeDL

from trackable import Trackable, FirstAppearance, DependentAppearance, LastDisappearance
from util import prettify_timestamp, get_next_run_from_srcom


def process_image(img_rgb, count, targets):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    threads = [
        threading.Thread(
            target=trackable.process,
            args=(img_gray, count,)
        )
        for trackable in targets.values()
    ]

    if not threads:
        return

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

def process_video(file_name):
    capture = cv2.VideoCapture(file_name)
    starting_ms = 0
    capture.set(cv2.CAP_PROP_POS_MSEC, starting_ms)
    duration = []
    targets: dict[str, Trackable] = {}
    video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    try:
        hades_door_ndarray = cv2.imread(f"templates/{video_height}p/hades_door.png", 0)
        reward_flare_ndarray = cv2.imread(f"templates/{video_height}p/reward_flare.png", 0)
        hades_title_ndarray = cv2.imread(f"templates/{video_height}p/hades_title.png", 0)
    except Exception as exc:
        print("Illegal resolution!")
        print(exc)
        raise

    targets["hades_door"] = FirstAppearance("hades_door", hades_door_ndarray)
    targets["start"] = DependentAppearance("start", reward_flare_ndarray, targets["hades_door"])
    targets["end"] = LastDisappearance("end", hades_title_ndarray, threshold=0.6)

    run_fps = capture.get(cv2.CAP_PROP_FPS)
    totalNoFrames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    count = run_fps/1000*starting_ms
    progress_bar = 0

    while True:
        grabbed, frame = capture.read()
        if not grabbed:
            break         # loop and a half construct is useful


        preprocess = time.time()
        process_image(frame, count, targets)
        postprocess = time.time()

        count += 1
        duration.append(postprocess-preprocess)

        if (new_progress := count*100 // totalNoFrames) > progress_bar+9:
            progress_bar = new_progress
            print(f"{progress_bar}% done, rolling average {1/mean(duration):.2f}fps ({1/mean(duration)/run_fps:.1f}x speed)")
            duration = list()

    # for name, trackable in targets.items():
    #     print(f"{name}: {trackable}")
    #     trackable.write_instances(Path("output"))

    run_duration = targets["end"].get_timestamp(run_fps)-targets["start"].get_timestamp(run_fps)

    print('\n'.join([
        f"Run analysis:",
        f"\tStart: {targets['start'].get_timestamp(run_fps, pretty=True)} (frame {targets['start'].triggered_at})",
        f"\tEnd: {targets['end'].get_timestamp(run_fps, pretty=True)} (frame {targets['end'].triggered_at})",
        f"\tDuration RTA: {prettify_timestamp(run_duration)}",
    ]))


# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("url", nargs='?', help = "URL to video")

# Read arguments from command line
args = parser.parse_args()

run_info = {
    "category": None,
    "players": list(),
    "url": args.url,
}

if not args.url:
    print("Getting most recent unverified run...")
    run = get_next_run_from_srcom()
    run_info["category"] = run["category"]["data"]["name"]
    for player in run["players"]["data"]:
        run_info["players"].append(player["names"]["international"])
    run_info["url"] = run["videos"]["links"][0]["uri"]

    print(f"Verifying {run_info['category']} run by {','.join(run_info['players'])}")

URLS = [run_info["url"]]
with tempfile.TemporaryDirectory() as output:
    with YoutubeDL(params={"paths": {"temp": output}, "post_hooks": [process_video], "format_sort": ["+res:480"], "quiet": True}) as ydl:
        ydl.download(URLS)
