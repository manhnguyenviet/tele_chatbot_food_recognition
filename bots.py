
from settings.settings import BARD_TOKEN, OPENAI_API_KEY


def get_answer_from_bard(question=""):
    if not question or not isinstance(question, str):
        return "Nothing to answer!"

    if not "?" in question:
        question = f"{question}?"

    from bardapi import Bard

    token = BARD_TOKEN
    bard = Bard(token=token)
    answer = bard.get_answer(question)
    return answer["content"]


def get_answer_from_claude(question=""):
    if not question or not isinstance(question, str):
        return "Nothing to answer!"

    if not "?" in question:
        question = f"{question}?"

    from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

    anthropic = Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        api_key="my api key",
    )

    completion = anthropic.completions.create(
        model="claude-2.1",
        max_tokens_to_sample=300,
        prompt=f"{HUMAN_PROMPT} {question}{AI_PROMPT}",
    )
    print(completion.completion)


def get_answer_from_chat_gpt(question=""):
    if not question or not isinstance(question, str):
        return "Nothing to answer!"

    if not "?" in question:
        question = f"{question}?"

    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a masterchef skilled in all European cuisines.",
            },
            {"role": "user", "content": question},
        ],
    )

    return completion.choices[0].message.content
