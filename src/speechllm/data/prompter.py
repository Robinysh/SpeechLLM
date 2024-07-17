# pylint: skip-file
import random

from speechllm.data.utils import clean_gigaspeech_tokens

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


class COTPrompter:
    def __init__(self, verbose: bool = False):
        self._verbose = verbose

    def generate_template(
        self,
        input_tokens,
        input_transcript=None,
        output_tokens=None,
        output_transcript=None,
    ) -> str:
        if output_tokens is None:
            prompt = f"{user_name}: {text_ins_sep} Step by step, give me the transcript of the provided audio, a chat response to the transcript, and read the response. {input_tokens} {user_end} {chatbot_name}: {response_sep}"
            return prompt
        input_transcript = clean_gigaspeech_tokens(input_transcript).lower()
        output_transcript = clean_gigaspeech_tokens(output_transcript).lower()

        prompt = f"{user_name}: {text_ins_sep} Step by step, give me the transcript of the provided audio, a chat response to the transcript, and read the response. {input_tokens} {user_end} {chatbot_name}: {response_sep} {input_transcript}\n {chatbot_name}: {output_transcript} {output_tokens} {chatbot_end}"
        return prompt


class SodaCOTPrompter:
    def __init__(self, context_range=(1, 99)):
        self.context_range = context_range

    def generate_template(self, context, response_tokens=None, output_transcript=None):
        num_contexts = random.randint(
            self.context_range[0], min(self.context_range[1], len(context))
        )
        dialogue = []
        for i, turn in enumerate(context):
            if (len(context) - i) % 2 == 0:
                header = f"{chatbot_name}:"
                footer = chatbot_end
            else:
                header = f"{user_name}:"
                footer = user_end
            dialogue.append(header + turn + footer)
        context = "\n".join(dialogue[max(len(context) - num_contexts, 0) :])
        if response_tokens is None or output_transcript is None:
            prompt = f"{user_name}: {text_ins_sep} Step by step, give me a chat response to the dialogue, and read the response.\n {context}\n[AnyGPT]: {response_sep}"
        else:
            prompt = f"{user_name}: {text_ins_sep} Step by step, give me a chat response to the dialogue, and read the response.\n {context}\n[AnyGPT]: {response_sep} {output_transcript} {response_tokens} {chatbot_end}"
        return prompt
