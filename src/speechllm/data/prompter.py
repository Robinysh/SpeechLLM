# pylint: skip-file

# It's just because the name MMGPT was used for model training in the early stages of research.
chatbot_name = "[AnyGPT]"
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

    def generate_template(self, input_tokens, output_tokens=None) -> str:
        if output_tokens is None:
            prompt = f"{user_name}: Let's chat.{input_tokens}{user_end} {chatbot_name}: <sosp>"
            return prompt

        prompt = f"{user_name}: Let's chat.{input_tokens}{user_end} {chatbot_name}: {output_tokens}{chatbot_end}"
        return prompt

    def generate_template_with_text_interface(
        self, input_tokens, output_tokens=None, output_transcript=None
    ):
        if output_tokens is None:
            prompt = f"{user_name}: Let's chat.{input_tokens}{user_end} {chatbot_name}: <sosp>"
            return prompt
        assert output_transcript is not None
        prompt = f"{user_name}: Let's chat.{input_tokens}{user_end} {chatbot_name}: {output_transcript} {output_tokens}{chatbot_end}"
        return prompt
