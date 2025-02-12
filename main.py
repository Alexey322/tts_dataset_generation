from utils import split_video_by_audio_chunks
from model import ASR


if __name__ == "__main__":
    asr_model = ASR()

    split_video_by_audio_chunks(asr_model=asr_model,
                                video_path=r"C:\Users\Alexey\Desktop/198855308-5bdbb260-526c-4d51-8bef-685c9a450a74.mp4",
                                save_dir='chunks',
                                min_chunk_duration=3,
                                max_chunk_duration=5,
                                min_time_between_words_for_separation=0.2,
                                min_chunk_word_score=0.5,
                                extra_time_for_chunk_borders=0.1)
    