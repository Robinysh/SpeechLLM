# pylint: skip-file

image_prefix = "👀"
speech_prefix = "🗣️"
music_prefix = "🎶"
audio_prefix = "👂"
start_of_image, end_of_image = "<soim>", "<eoim>"
start_of_speech, end_of_speech = "<sosp>", "<eosp>"
start_of_music, end_of_music = "<somu>", "<eomu>"
start_of_audio, end_of_audio = "<soau>", "<eoau>"
image_vocab_size = 8192
speech_vocab_size = 1024
music_codebook_size = 2048
music_codebook_num = 4
music_vocab_size = music_codebook_size * music_codebook_num
audio_codebook_size = 1024
audio_codebook_num = 4
audio_vocab_size = audio_codebook_size * audio_codebook_num


modal_special_str = {
    "image": {
        "prefix": image_prefix,
        "sos": start_of_image,
        "eos": end_of_image,
        "vocab_size": image_vocab_size,
    },
    "speech": {
        "prefix": speech_prefix,
        "sos": start_of_speech,
        "eos": end_of_speech,
        "vocab_size": speech_vocab_size,
    },
    "music": {
        "prefix": music_prefix,
        "sos": start_of_music,
        "eos": end_of_music,
        "vocab_size": music_vocab_size,
    },
}
