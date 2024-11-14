"""Common functions for operations on a standardized dataset"""
import random

BRIEF_PROMPTS = {
    'default': "Answer the following question as briefly as possible.\n",
    'chat': 'Answer the following question in a single brief but complete sentence.\n'}


def sample_multiple_entries(dataset, n=1):
    """Randomly select and return `n` entries from the dataset."""
    if not dataset:
        raise ValueError("The dataset is empty, cannot sample entries.")
    if n > len(dataset):
        raise ValueError(f"Requested {n} samples, but the dataset only has {len(dataset)} entries.")
    return random.sample(dataset, n)



def make_prompt(context, question, choices='', answer, brief, brief_always = False,use_context= False,use_choices = False):
        prompt = ''
        if brief_always:
            prompt += brief
        if use_context and (context is not None):
            prompt += f"Context: {context}\n"
        prompt += f"Question: {question}\n"

        if use_choices and (choices is not None):
            prompt += f"Choices: {choices}\n"

        if answer:
            prompt += f"Answer: {answer}\n\n"
        else:
            prompt += 'Answer:'
        return prompt

def construct_fewshot_prompt_from_indices(dataset, example_indices, brief, make_prompt,brief_always,use_context,use_choices):
    """Given a dataset and indices, construct a fewshot prompt."""
    if not brief_always:
        prompt = brief
    else:
        prompt = ''

    for example_index in example_indices:

        example = dataset[example_index]
        question = example["question"]
        answer = example["answers"]["text"]

        context = example.get("context", "")
        choices = example.get("choices", "")

        prompt = prompt + make_prompt(context,question,choices,answer, brief, brief_always,use_context,use_choices)

    return prompt