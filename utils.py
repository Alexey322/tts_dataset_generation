import os.path

import librosa

import subprocess
import hashlib
import soundfile as sf


def convert_to_wav(file_path, save_path):
    cmd = f'ffmpeg -i "{file_path}" -vn -y -acodec pcm_s16le -ac 1 "{save_path}"'

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    if os.path.exists(save_path):
        return True
    else:
        return False


def load_audio_from_video(video_path):
    file_hash = hashlib.md5(open(video_path, 'rb').read()).hexdigest()
    audio_path = f'cached/{file_hash}.wav'

    if not os.path.exists(audio_path):
        os.makedirs('cached', exist_ok=True)
        result = convert_to_wav(video_path, audio_path)

    audio, sr = librosa.load(audio_path, sr=None)

    return audio, sr


def get_chunk_start(words_timestamps, start_search_ind, min_time_between_words_for_separation):
    chunk_start_ind = -1

    if start_search_ind == 0:
        if words_timestamps[0]['start'] >= min_time_between_words_for_separation:
            chunk_start_ind = 0
        else:
            start_search_ind += 1

    if chunk_start_ind == -1:
        words_timestamps = words_timestamps[start_search_ind-1:]

        for ind in range(1, len(words_timestamps)):
            last_word = words_timestamps[ind-1]
            current_word = words_timestamps[ind]

            if current_word['start'] - last_word['end'] >= min_time_between_words_for_separation:
                chunk_start_ind = ind + start_search_ind - 1
                break

    return chunk_start_ind



def get_chunk_best_end(words_timestamps, start_search_ind, min_chunk_duration, max_chunk_duration):
    words_timestamps = words_timestamps[start_search_ind:]

    chunk_start = words_timestamps[0]['start']

    if words_timestamps[-1]['end'] - chunk_start >= min_chunk_duration:
        best_max_time_between_words = 0
        best_word_ind = 0

        worst_word_score = 1.0
        worst_word_score_ind = 0

        for ind in range(len(words_timestamps)):
            current_word = words_timestamps[ind]

            if current_word['end'] - chunk_start > max_chunk_duration:
                break

            if current_word['score'] <= worst_word_score:
                worst_word_score = current_word['score']
                worst_word_score_ind = ind

            if current_word['end'] - chunk_start >= min_chunk_duration:
                if ind != len(words_timestamps) - 1:
                    next_word = words_timestamps[ind + 1]

                    time_to_next_word = next_word['start'] - current_word['end']

                    if time_to_next_word > best_max_time_between_words:
                        best_max_time_between_words = time_to_next_word
                        best_word_ind = ind
                else:
                    best_max_time_between_words = 9999999
                    best_word_ind = ind

        best_word_ind += start_search_ind
        worst_word_score_ind += start_search_ind
    else:
        best_max_time_between_words = -1
        best_word_ind = -1

        worst_word_score = -1
        worst_word_score_ind = -1


    return best_max_time_between_words, best_word_ind, worst_word_score, worst_word_score_ind


def merge_words_by_time(words_timestamps, min_chunk_duration, max_chunk_duration,
                        min_time_between_words_for_separation=0.0, min_chunk_word_score=0.0,
                        extra_time_for_chunk_borders=0.0):
    merged_words = []
    current_timestamp_ind = 0

    while current_timestamp_ind != len(words_timestamps):
        chunk_start_ind = get_chunk_start(words_timestamps, current_timestamp_ind, min_time_between_words_for_separation)

        # there is no such duration between the remaining words
        if chunk_start_ind == -1:
            break

        current_timestamp_ind = chunk_start_ind

        chunk_info = get_chunk_best_end(words_timestamps, current_timestamp_ind, min_chunk_duration,
                                        max_chunk_duration)
        time_between_words, end_word_ind, worst_word_score, worst_word_score_ind = chunk_info

        # the remaining chunk is so small that we cannot get the required min duration
        if end_word_ind == -1:
            break
        # impossible to reach the minimum duration and the next word end > max_chunk_duration
        elif current_timestamp_ind == end_word_ind:
            current_timestamp_ind += 1
            continue

        if worst_word_score >= min_chunk_word_score:
            if time_between_words >= min_time_between_words_for_separation:
                tmp_words = ' '.join([word['word'] for word in words_timestamps[current_timestamp_ind: end_word_ind+1]])
                start = words_timestamps[current_timestamp_ind]['start'] - extra_time_for_chunk_borders
                end = words_timestamps[end_word_ind]['end'] + extra_time_for_chunk_borders

                merged_words.append({'words': tmp_words,
                                     'start': start,
                                     'end': end})

                current_timestamp_ind = end_word_ind + 1
            else:
                current_timestamp_ind += 1
        else:
            current_timestamp_ind += 1

    return merged_words


def split_video_by_audio_chunks(asr_model, video_path, save_dir, min_chunk_duration, max_chunk_duration,
                                min_time_between_words_for_separation=0.0, min_chunk_word_score=0.0,
                                extra_time_for_chunk_borders=0.0):
    """
    Разбивает входной видеофайл на фрагменты длиной [min_chunk_duration, max_chunk_duration]

    Если min_time_between_words_for_separation или min_chunk_word_score не равны нулю, то разбиваться будут
    только те фрагменты, которые удовлетворяют этим параметрам

    Сохраняет фрагменты с текстовой аннотацей в папку save_dir

    :param asr_model: Инстанс модели для распознавания речи
    :param video_path: Путь к видеофайлу
    :param save_dir: Путь к директории для сохранения фрагментов
    :param min_chunk_duration: Минимальная длина фрагмента
    :param max_chunk_duration: Максимальная длина фрагмента
    :param min_time_between_words_for_separation: Минимальное время между началом фрагмента с предыдущим словом и концом
    фрагмента со следующим словом
    :param min_chunk_word_score: Минимальный скор слова в разбиваемом фрагменте
    :param extra_time_for_chunk_borders: Увеличивает границы фрагмента, что бы аудио не заканчивалось и не начиналось
    резко. Не может быть больше чем min_time_between_words_for_separation / 2.
    :return:
    """
    result = asr_model.transcribe(video_path)

    words_timestamps = []

    for segment in result['segments']:
        for word in segment['words']:
            words_timestamps.append({'word': word['word'], 'start': word['start'], 'end': word['end'],
                                     'score': word['score']})

    extra_time_for_chunk_borders = min(min_time_between_words_for_separation / 2, extra_time_for_chunk_borders)

    # Since we add extra time for each chunk, the final chunk length will be in the range
    # [min_chunk_duration + 2 * extra_time_for_chunk_borders; max_chunk_duration + 2 * extra_time_for_chunk_borders].
    # So we have to subtract the extra time to get the correct range.
    min_chunk_duration -= 2 * extra_time_for_chunk_borders
    max_chunk_duration -= 2 * extra_time_for_chunk_borders

    merged_words_list = merge_words_by_time(words_timestamps, min_chunk_duration, max_chunk_duration,
                                            min_time_between_words_for_separation, min_chunk_word_score,
                                            extra_time_for_chunk_borders)

    audio, sr = load_audio_from_video(video_path)

    os.makedirs(save_dir, exist_ok=True)

    for ind, merged_words in enumerate(merged_words_list):
        audio_chunk_start = round(merged_words['start'] * sr)
        audio_chunk_end = round(merged_words['end'] * sr)

        save_chunk_audio_path = f"{save_dir}/{ind + 1}.wav"
        save_chunk_text_path = f"{save_dir}/{ind + 1}.txt"
        
        sf.write(save_chunk_audio_path, audio[audio_chunk_start:audio_chunk_end], int(sr))

        with open(save_chunk_text_path, 'w', encoding='utf-8') as file:
            file.write(merged_words['words'])
