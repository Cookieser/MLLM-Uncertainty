"""Load and standardize the dataset"""

"""
Transform the dataset to the following structure:
{
    "question": "what is the capital of france?",
    "answers": {
        "text": ["paris"],
        "answer_start": [0]  # Optional, mainly used for datasets with context
    },
    "context": "france is a country in europe with several major cities.",
    "id": "12345",
    "options": ["paris", "london", "berlin", "madrid"]  # Optional attribute
}

Attributes to include:
- question
- answers
- context
- id
- options (optional)

Ensure all essential attribute names are standardized to lowercase. Additional attributes can remain unchanged.
"""

import os
import json
import hashlib
import datasets


def load_ds(dataset_name, seed):
    train_dataset, validation_dataset = None, None
    if dataset_name == "squad":
        dataset = datasets.load_dataset("squad_v2")
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]

    elif dataset_name == 'svamp':
        dataset = datasets.load_dataset('ChilleD/SVAMP')
        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]

        reformat = lambda x: {
            'question': x['Question'], 
            'context': x['Body'], 
            'type': x['Type'],
            'equation': x['Equation'], 
            'id': x['ID'],
            'answers': {'text': [str(x['Answer'])]}}

        train_dataset = [reformat(d) for d in train_dataset]
        validation_dataset = [reformat(d) for d in validation_dataset]

    elif dataset_name == 'nq':
        dataset = datasets.load_dataset("nq_open")
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]
        md5hash = lambda s: str(int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16))

        reformat = lambda x: {
            'question': x['question']+'?',
            'answers': {'text': x['answer']},
            'context': '',
            'id': md5hash(str(x['question'])),
        }

        train_dataset = [reformat(d) for d in train_dataset]
        validation_dataset = [reformat(d) for d in validation_dataset]

    elif dataset_name == "trivia_qa":
        dataset = datasets.load_dataset('TimoImhof/TriviaQA-in-SQuAD-format')['unmodified']
        dataset = dataset.train_test_split(test_size=0.2, seed=seed)
        train_dataset = dataset['train']
        validation_dataset = dataset['test']

    # have some choices
    elif dataset_name == 'trustful_qa_mc1':
        dataset = datasets.load_dataset("truthfulqa/truthful_qa",'multiple_choice')
        validation_dataset = dataset["validation"]


        md5hash = lambda s: str(int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16))

        reformat = lambda x: {
            'question': x['question'],
            'choices': x['mc1_targets']['choices'],
            'answers': {'text': x['mc1_targets']['labels']},
            'id': md5hash(str(x['question'])),
        }

        validation_dataset = [reformat(d) for d in validation_dataset]

    elif dataset_name == 'trustful_qa_mc2':
        dataset = datasets.load_dataset("truthfulqa/truthful_qa",'multiple_choice')
        validation_dataset = dataset["validation"]


        md5hash = lambda s: str(int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16))

        reformat = lambda x: {
            'question': x['question'],
            'choices': x['mc2_targets']['choices'],
            'answers': {'text': x['mc2_targets']['labels']},
            'id': md5hash(str(x['question'])),
        }

        validation_dataset = [reformat(d) for d in validation_dataset]


    else:
        raise ValueError


    return train_dataset, validation_dataset




