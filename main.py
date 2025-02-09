from utils import split_video_by_audio_chunks
from model import ASR


if __name__ == "__main__":
    asr_model = ASR()

    split_video_by_audio_chunks(asr_model=asr_model,
                                video_path=f"IELTS Speaking Topics and Sample Answers.mp4.wav",
                                save_dir='chunks',
                                min_chunk_duration=3,
                                max_chunk_duration=5,
                                min_time_between_words_for_separation=0.1,
                                min_chunk_word_score=0.5)
    