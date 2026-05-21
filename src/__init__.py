from dotenv import load_dotenv
import os

load_dotenv()

ffmpeg_path = os.getenv("FFMPEG_PATH")
if ffmpeg_path and os.path.exists(ffmpeg_path):
    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path

os.makedirs("output", exist_ok=True)
os.makedirs("output/audio", exist_ok=True)
os.makedirs("output/frames", exist_ok=True)