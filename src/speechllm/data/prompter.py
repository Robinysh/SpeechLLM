# pylint: skip-file
import random

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


"""
def drop_tokens(item):
    shm = SharedMemory(
        create=False, size=4, name=f"global_step_{os.environ['MASTER_PORT']}"
    )
    arr = np.ndarray([1], np.int32, shm.buf)
    global_step = arr[0]
    mean = max(min(global_step / 50000, 0.999), 0.001)
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
"""


def drop_tokens(item, global_step=None):
    """
    if global_step is None:
        shm = SharedMemory(
            create=False, size=4, name=f"global_step_{os.environ['MASTER_PORT']}"
        )
        arr = np.ndarray([1], np.int32, shm.buf)
        global_step = arr[0]
    """
    return item
    # drop_percentile = max(min(global_step / 100000, 1), 0)
    tokens = item.split(" ")
    if global_step is None:
        # drop_percentile = max(min(14000 / 100000, 1), 0)
        min_drop_num = 2
    else:
        # drop_percentile = max(min(global_step / 100000, 1), 0)
        min_drop_num = int(len(tokens))
    # drop_percentile = max(min((global_step - 50000 + 30000) / 50000, 1), 0)
    # drop_percentile = max(min((66000 - 50000 + 30000) / 50000, 1), 0)
    # drop_percentile = max(min((70000 - 50000 + 30000) / 50000, 1), 0)
    # drop_percentile = max(min(10000 / 25000 + (global_step + 50000) / 200000, 1), 0)
    # drop_percentile = max(min(10000 / 25000, 1), 0)
    # min_drop_num = global_step // 5000
    drop_num = min(
        min_drop_num + int(np.random.exponential(scale=0.25)), int(len(tokens))
    )
    # drop_num = int(len(tokens))
    # return " ".join(tokens[drop_num:])
    if drop_num != 0:
        tokens[:drop_num] = ["—"] * drop_num  # replace tokens with unused token
        # tokens = tokens[:-drop_num]
        # tokens = tokens[drop_num:]
    return " ".join(tokens)


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
        instruction = f"You are {chatbot_name}. You are chatting with {user_name}. Step by step, give me the transcript of the provided audio, a chat response to the transcript, and read the response."

        if output_tokens is None:
            prompt = f"{instruction} {text_ins_sep} {user_name}: {input_tokens} {user_end} {chatbot_name}: {response_sep}"
            return prompt
        input_transcript = clean_gigaspeech_tokens(input_transcript).lower()
        output_transcript = clean_gigaspeech_tokens(output_transcript).lower()

        prompt = f"{instruction} {text_ins_sep} {user_name}: {input_tokens} {user_end} {chatbot_name}: {response_sep} {input_transcript}\n{chatbot_name}: {output_transcript} {output_tokens} {chatbot_end}"
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


class SodaASRTTSCOTPrompter:
    def __init__(self, context_range=(1, 99)):
        self.context_range = context_range

    def generate_template(
        self,
        audio_tokens,
        context=None,
        context_interval=None,
        inference=False,
        teacher=False,
    ):
        audio_tokens = audio_tokens[context_interval[0] : context_interval[1]]
        context = context[context_interval[0] : context_interval[1]]

        dialogue = []
        for i, (turn_token, turn_context) in enumerate(zip(audio_tokens, context)):
            if inference and i == len(context) - 1:
                break
            if (len(context) - i) % 2 == 1:
                header = f"{chatbot_name}:"
                footer = chatbot_end
                if teacher:
                    dialogue.append(
                        f"{header} {response_sep} {context[i - 1]}\n{chatbot_name}: {turn_context} <sosp>"
                    )
                    continue
                if i > 0:
                    dialogue.append(
                        f"{header} {response_sep} {context[i - 1]}\n{chatbot_name}: {turn_context} {turn_token} {footer}"
                    )
                else:
                    dialogue.append(
                        f"{header} {response_sep} {turn_context} {turn_token} {footer}"
                    )
            else:
                header = f"{user_name}:"
                footer = user_end
                dialogue.append(f"{header} {turn_token} {footer}")
        dialogue = "\n".join(dialogue)

        instruction = (
            f"You are {chatbot_name}. You are chatting with {user_name}. "
            + "Step by step, give me the transcript of the provided audio, a chat response to the transcript, and read the response."
        )
        prompt = f"{instruction} {text_ins_sep} {dialogue}"
        if inference:
            prompt = f"{prompt}\n[AnyGPT]: {response_sep}"
        return prompt

    def generate_implicit_template(
        self,
        audio_tokens,
        context=None,
        context_interval=None,
        inference=False,
    ):
        audio_tokens = audio_tokens[context_interval[0] : context_interval[1]]
        context = context[context_interval[0] : context_interval[1]]

        dialogue = []
        for i, (turn_token, turn_context) in enumerate(zip(audio_tokens, context)):
            if inference and i == len(context) - 1:
                break
            if (len(context) - i) % 2 == 1:
                header = f"{chatbot_name}:"
                footer = chatbot_end
                if i > 0:
                    dialogue.append(
                        # f"{header} {response_sep} {drop_tokens(context[i - 1])}\n{chatbot_name}: {drop_tokens(turn_context)} {turn_token} {footer}"
                        # f"{header} {response_sep} {drop_tokens(context[i - 1])}\n{chatbot_name}: {turn_context} {turn_token} {footer}"
                        f"{header} {response_sep} {drop_tokens(context[i - 1], global_step=9999999)}\n{chatbot_name}: {drop_tokens(turn_context)} {turn_token} {footer}"
                    )
                else:
                    dialogue.append(
                        # f"{header} {response_sep} {drop_tokens(turn_context)} {turn_token} {footer}"
                        f"{header} {response_sep} {drop_tokens(turn_context)} {turn_token} {footer}"
                    )
            else:
                header = f"{user_name}:"
                footer = user_end
                dialogue.append(f"{header} {turn_token} {footer}")
        dialogue = "\n".join(dialogue)

        instruction = (
            f"You are {chatbot_name}. You are chatting with {user_name}. "
            # + drop_tokens(
            + (
                "Step by step, give me the transcript of the provided audio, a chat response to the transcript, and read the response."
            )
        )
        prompt = f"{instruction} {text_ins_sep} {dialogue}"
        if inference:
            prompt = f"{prompt}\n[AnyGPT]: {response_sep}"
        return prompt
