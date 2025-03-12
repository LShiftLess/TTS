import traceback
import librosa
import soundfile as sf
from pytubefix import YouTube
from pytubefix.cli import on_progress

def download_and_denoise_youtube_audio(youtube_url, output_file, noise_reduce_factor=2.0):
    """
    Downloads audio from YouTube, removes noise, and saves the denoised audio.

    Args:
        youtube_url (str): The URL of the YouTube video.
        noise_reduce_factor (float): Strength of noise reduction (adjust based on the audio).
    """

    try:
        yt = YouTube(youtube_url, on_progress_callback=on_progress)
        audio_stream = yt.streams.get_audio_only()
        downloaded_file = audio_stream.download(filename="temp_audio.m4a")
        print(f"Audio downloaded to: {downloaded_file}")

        y, sr = librosa.load("temp_audio.m4a")
        # y = nr.reduce_noise(y=y, sr=sr, stationary=False, prop_decrease=noise_reduce_factor)
        sf.write(output_file, y, sr)
        print(f"Converted m4a file into wav file")

    except Exception as e:
        traceback.print_exc()
    finally:  # Clean up the temporary file
        import os

        try:
            os.remove("temp_audio.m4a")
            print("Temporary file removed.")
        except FileNotFoundError:
            pass  # It's okay if the file wasn't created


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--url", dest="url", required=True)
    parser.add_argument("-o", "--output", dest="output", default="output.wav")
    args = parser.parse_args()

    url_download = args.url
    output = args.output
    if url_download is None:
        raise ValueError("Providing an url for downloading is necessary!")

    download_and_denoise_youtube_audio(url_download, output)  # Uses default output file and noise_reduce_factor


if __name__ == "__main__":
    main()
