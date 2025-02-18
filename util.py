import timeit
import whisper
from typing import Tuple
import matplotlib.pyplot as plt
import subprocess

def load_model(model_name: str = "tiny.en", ff: bool = False, cut_region=None) -> whisper.Whisper:
    return whisper.load_model(model_name, ext_feature_flag=ff, cut_region=cut_region)


def transcribe(model: whisper.Whisper, audio_path: str) -> Tuple[str, float]:
    start_time = timeit.default_timer()
    transcription = model.transcribe(audio_path)
    transcription = transcription.get("text", "")
    elapsed_time = timeit.default_timer() - start_time
    return transcription, elapsed_time


def calculate_wer(hypothesis: str, reference: str) -> float:
    hyp_words = hypothesis.strip().lower().split()
    ref_words = reference.strip().lower().split()

    if not ref_words:
        return float("inf") if hyp_words else 0.0

    dp = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

    for i in range(len(ref_words) + 1):
        dp[i][0] = i
    for j in range(len(hyp_words) + 1):
        dp[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,       # deletion
                    dp[i][j - 1] + 1,       # insertion
                    dp[i - 1][j - 1] + 1,   # substitution
                )

    return dp[len(ref_words)][len(hyp_words)] / len(ref_words)

# grab audio length of file ( based on specified path ) in ms
def get_audio_len( file_path ):
    command = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", file_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return float(result.stdout.strip()) * 1000
