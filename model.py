import whisperx


class Aligner:
    def __init__(self, lang):
        self.align_model, self.metadata = whisperx.load_align_model(language_code=lang, device='cuda')

    def align(self, segments, audio):
        result = whisperx.align(segments, self.align_model, self.metadata, audio, device='cuda',
                                return_char_alignments=False)

        return result


class ASR:
    def __init__(self):
        self.whisperx_model = whisperx.load_model(whisper_arch="medium.en", device='cuda',
                                                  download_root=f'whisper_model',
                                                  asr_options={'suppress_numerals': True})
        self.aligner = Aligner('en')

    def transcribe(self, wav_path):
        audio = whisperx.load_audio(wav_path)
        result = self.whisperx_model.transcribe(audio, batch_size=1)

        result = whisperx.align(result["segments"], self.aligner.align_model, self.aligner.metadata, audio,
                                device='cuda', return_char_alignments=False)

        return result

