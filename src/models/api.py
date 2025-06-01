import sys
import time
from openai import OpenAI
from typing import Union, List

class LLM_API:
    def __init__(self, model_name, api_key, base_url):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model_name

    def respond(self, messages, temperature=1, max_tokens=256, stop=None):

        repeat_num = 0
        response_data = None
        while response_data == None:
            repeat_num += 1
            if repeat_num>5:
                response_data = "I Don't Know!"
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages= messages,
                    timeout=180,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=0.9,
                    stop=stop
                )
                response_data = completion.choices[0].message.content
                break
            except KeyboardInterrupt:
                sys.exit()
            except:
                print("Request timed out, retrying...")
                
        return response_data


def gpt_call(client, query):
    if isinstance(query, List):
        messages = query
    elif isinstance(query, str):
        messages = [{"role": "system", "content": "You are a helpful assistant!"},
                    {"role": "user", "content": query}]
    response = client.respond(messages)

    return response

def gpt_call_append(client, dialog_hist, query: str):
    dialog_hist.append({"role": "user", "content": query})
    response = gpt_call(client, dialog_hist)
    dialog_hist.append({"role": "assistant", "content": response})
    return response, dialog_hist