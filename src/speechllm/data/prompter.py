# pylint: skip-file
import random
from multiprocessing.shared_memory import SharedMemory

import numpy as np

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


def drop_tokens(item):
    shm = SharedMemory(create=False, size=4, name="global_step")
    arr = np.ndarray([1], np.int32, shm.buf)
    global_step = arr[0]
    mean = max(min(global_step / 15000, 0.999), 0.001)
    sd = 0.1
    n = mean * (1 - mean) / sd**2
    a = mean * n
    b = (1 - mean) * n
    drop_ratio = np.random.beta(a, b, 1)
    tokens = item.split(" ")
    mean_num_tokens = int(len(tokens) * mean)
    num_tokens = min(
        max(int(len(tokens) * drop_ratio), mean_num_tokens - 5), mean_num_tokens + 5
    )
    return " ".join(tokens[num_tokens:])


class Prompter:
    def __init__(self, verbose: bool = False):
        self._verbose = verbose

    def generate_template(
        self,
        input_tokens,
        output_tokens=None,
    ) -> str:
        instruction = f"You are {chatbot_name}. You are chatting with {user_name}. "
        if output_tokens is None:
            prompt = f"{instruction} {text_ins_sep} {user_name}: {input_tokens} {user_end} {chatbot_name}: {response_sep}"
            return prompt
        prompt = f"{instruction} {text_ins_sep} {user_name}: {input_tokens} {user_end} {chatbot_name}: {response_sep} {output_tokens} {chatbot_end}"
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
        instruction = "Step by step, give me the transcript of the provided audio, a chat response to the transcript, and read the response."

        if output_tokens is None:
            prompt = f"{instruction} {text_ins_sep} {user_name}: {input_tokens} {user_end} {chatbot_name}: {response_sep}"
            return prompt
        input_transcript = clean_gigaspeech_tokens(input_transcript).lower()
        output_transcript = clean_gigaspeech_tokens(output_transcript).lower()

        prompt = f"{instruction} {text_ins_sep} {user_name}: {input_tokens} {user_end} {chatbot_name}: {response_sep} {input_transcript}\n {chatbot_name}: {output_transcript} {output_tokens} {chatbot_end}"
        return prompt

    def generate_implicit_template(
        self,
        input_tokens,
        input_transcript=None,
        output_tokens=None,
        output_transcript=None,
    ) -> str:
        instruction = (
            f"You are {chatbot_name}. You are chatting with {user_name}. "
            + drop_tokens(
                "Step by step, give me the transcript of the provided audio, a chat response to the transcript, and read the response."
            )
        )
        if output_tokens is None:
            prompt = f"{instruction} {text_ins_sep} {user_name}: {input_tokens} {user_end} {chatbot_name}: {response_sep}"
            return prompt
        input_transcript = drop_tokens(
            clean_gigaspeech_tokens(input_transcript).lower()
        )
        output_transcript = drop_tokens(
            clean_gigaspeech_tokens(output_transcript).lower()
        )

        prompt = f"{instruction} {text_ins_sep} {user_name}: {input_tokens} {user_end} {chatbot_name}: {response_sep} {input_transcript}\n {chatbot_name}: {output_transcript} {output_tokens} {chatbot_end}"
        return prompt


class SodaPrompter:
    def __init__(self, context_range=(1, 99)):
        self.context_range = context_range

    def generate_template(self, context, response_tokens=None):
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

        instruction = f"You are {chatbot_name}. You are chatting with {user_name}."
        if response_tokens is None:
            prompt = f"{instruction} {text_ins_sep} {context}\n[AnyGPT]: {response_sep}"
        else:
            prompt = f"{instruction} {text_ins_sep} {context}\n[AnyGPT]: {response_sep} {response_tokens} {chatbot_end}"

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

        instruction = "Step by step, give me a chat response to the dialogue, and read the response."
        if response_tokens is None or output_transcript is None:
            prompt = f"{instruction} {text_ins_sep} {context}\n[AnyGPT]: {response_sep}"
        else:
            prompt = f"{instruction} {text_ins_sep} {context}\n[AnyGPT]: {response_sep} {output_transcript} {response_tokens} {chatbot_end}"

        return prompt

    def generate_implicit_template(
        self, context, response_tokens=None, output_transcript=None
    ):
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

        instruction = (
            f"You are {chatbot_name}. You are chatting with {user_name}. "
            + drop_tokens(
                "Step by step, give me a chat response to the dialogue, and read the response."
            )
        )
        if response_tokens is None or output_transcript is None:
            prompt = f"{instruction} {text_ins_sep} {context}\n[AnyGPT]: {response_sep}"
        else:
            prompt = f"{instruction} {text_ins_sep} {context}\n[AnyGPT]: {response_sep} {drop_tokens(output_transcript)} {response_tokens} {chatbot_end}"
        return prompt
