import tiktoken
import openai


MODEL_ID = "gpt-3.5-turbo-0301"
COST_PER_TOKEN = 0.002 / 1000  # 0.002 USD per 1000 tokens


def num_tokens_from_messages(messages, model=MODEL_ID):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")


def compute_message_cost(messages, model=MODEL_ID, cost_per_token=COST_PER_TOKEN):
    """Returns the cost of a list of messages."""
    return num_tokens_from_messages(messages, model) * cost_per_token


def create_message(role, content):
    """Returns a message dictionary."""
    assert role in ["assistant", "user", "system"]

    return {"role": role, "content": content}

def query_model(messages, model_id=MODEL_ID):
    """Returns the response from the model."""
    response = openai.ChatCompletion.create(
        model=model_id,
        messages=messages
        )
    return response
