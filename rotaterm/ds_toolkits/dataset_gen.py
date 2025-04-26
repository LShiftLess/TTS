import os
import csv
import re
from tqdm import tqdm
from num2words import num2words
import whisperx
import torch
import torchaudio
from pydub import AudioSegment

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
TARGET_SR = 22050 # Target sample rate (22kHz)


def split_audio(input_file, output_dir, chunk_length_ms):
    """
    Split the input audio into chunks of specified length.
    Returns a list of file paths for the chunks.
    """

    print("Splitting formatted audio into chunks...")
    audio: AudioSegment = AudioSegment.from_wav(input_file)

    chunks = []
    for i in tqdm(range(chunk_length_ms, len(audio), chunk_length_ms), desc='split & format'):
        chunk = audio[i - chunk_length_ms : i]
        
        # Format the audio here because the frames of the raw audio is not a whole number. That will cause the error.
        chunk = chunk.set_channels(1)
        chunk = chunk.set_frame_rate(TARGET_SR)

        chunk_name = f"chunk_{((i // chunk_length_ms) - 1):04d}.wav"
        chunk_path = os.path.join(output_dir, chunk_name)

        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    print(f"Created {len(chunks)} chunks.")
    return chunks


def transcribe_with_whisperx(audio_path, model):
    """
    Transcribe a single audio file using WhisperX.
    Returns the transcription text.
    """

    try:
        # Load audio
        audio, sr = torchaudio.load(audio_path)
        if len(audio.shape) > 1:  # Convert stereo to mono if needed
            audio = audio.mean(dim=0)

        # Transcribe
        print(f"Transcribing {audio_path}...")
        result = model.transcribe(audio.numpy(), batch_size=16)
        transcription = " ".join(segment["text"] for segment in result["segments"]).strip()

        if not transcription:
            transcription = "[No speech detected]"
        return transcription
    except Exception as e:
        print(f"Error transcribing {audio_path}: {e}")
        return "[Transcription failed]"


def normalize_transcription(text):
    """
    Normalize transcription for TTS:
    - Convert numbers to words (e.g., "123" to "one hundred twenty-three").
    - Convert to lowercase.
    - Remove unwanted punctuation (keep periods and commas for sentence structure).
    - Handle edge cases like empty or failed transcriptions.
    Returns the normalized text.
    """
    if not text or text in ["[No speech detected]", "[Transcription failed]"]:
        return text  # Return unchanged if it's a placeholder or empty

    # Step 1: Convert numbers to words
    def replace_numbers(match: re.Match):
        num = match.group(0)
        try:
            # Convert the number to words
            return num2words(int(num))
        except (ValueError, OverflowError):
            return num  # If conversion fails, return the original string

    # Find all numbers in the text using regex and replace them
    text_with_numbers_converted = re.sub(r"\b\d+\b", replace_numbers, text)

    # Step 2: Convert to lowercase
    text_lowercase = text_with_numbers_converted.lower()

    # Step 3: Remove unwanted punctuation (keep periods and commas)
    # Replace multiple spaces with single space, strip leading/trailing spaces
    text_cleaned = re.sub(r"[^\w\s.,]", "", text_lowercase)  # Keep alphanumeric, spaces, periods, commas
    text_cleaned = re.sub(r"\s+", " ", text_cleaned).strip()

    return text_cleaned


def generate_metadata(chunks, model_name, language, output_dir):
    """
    Generate metadata.csv from the chunks and their transcriptions.
    Format: file_name|transcription|normalized_transcription
    """
    metadata_path = os.path.join(output_dir, "metadata.csv")
    print(f"Generating {metadata_path}...")

    # Load WhisperX model
    print(f"Loading WhisperX model: {model_name}...")
    model = whisperx.load_model(model_name, DEVICE, language=language)

    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        for chunk_path in tqdm(chunks, desc="transcriptions"):
            # Get the base name (without extension or path)
            base_name = os.path.splitext(os.path.basename(chunk_path))[0]

            # Transcribe the chunk
            transcription = transcribe_with_whisperx(chunk_path, model)

            # Normalize the transcription
            normalized = normalize_transcription(transcription)

            # Write to CSV
            writer.writerow([base_name, transcription, normalized])
            print(f"Processed {base_name}: {transcription}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Path to a wav file", required=True)
    parser.add_argument(
        "-c", "--chunk_len", dest="chunk_len", type=int, default=10000, help="Length of each chunk in milliseconds"
    )
    parser.add_argument(
        "-l", "--lang", "--language", dest="language", default="en", help='Language code (e.g., "en" for English)'
    )
    parser.add_argument(
        "-m",
        "--model",
        "--model_arch",
        dest="model",
        default="large",
        help="WhisperX model: tiny, base, small, medium, large",
    )
    args = parser.parse_args()

    input_file = args.input
    chunk_len = args.chunk_len
    model = args.model
    lang = args.language

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input audio file {input_file} not found!")

    output_dir = os.path.join(os.path.splitext(input_file)[0])
    sub_dir = os.path.join(output_dir, "wavs")

    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(sub_dir, exist_ok=True)

    # Step 1: Split audio into chunks
    chunks = split_audio(input_file, sub_dir, chunk_len)

    # Step 2: Generate metadata with transcriptions
    generate_metadata(chunks, model, lang, output_dir)        

    print("Done! Dataset generated in", output_dir)


if __name__ == "__main__":
    main()
