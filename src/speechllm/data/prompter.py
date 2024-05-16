# pylint: skip-file

# It's just because the name MMGPT was used for model training in the early stages of research.
chatbot_name = "[MMGPT]"
user_name = "[Human]"
user_end = "<eoh>"
chatbot_end = "<eos>"
speech_response_sep = "<eot>"
text_ins_sep = "<-Ins->"
response_sep = "<-Res->"
special_tokens = [
    user_name,
    chatbot_name,
    user_end,
    chatbot_end,
    response_sep,
    text_ins_sep,
]

system_prompt = "You are an AI assistant named MMGPT who can understand and generate multimodal content, including text, speech, images and audio."


class Prompter:
    def __init__(self, verbose: bool = False):
        self._verbose = verbose

    def generate_template(self, input_tokens, output_tokens) -> str:
        prompt = (
            user_name
            + f": {input_tokens}"
            + f"{user_end} {chatbot_name}: "
            + f"{output_tokens}"
            + f"{chatbot_end}"
        )
        return prompt
