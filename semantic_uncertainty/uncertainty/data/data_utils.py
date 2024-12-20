"""Data Loading Utilities."""
import os
import json
import hashlib
import datasets
import pandas as pd
from tqdm import tqdm

mmvp_image_path = "/work/images/images"
mmvp_original_image_path = "/work/images/images/mmvp"
mmvp_dataset_path = "semantic_uncertainty/uncertainty/data/Questions.csv" 

VQA_dataset_path = "semantic_uncertainty/uncertainty/data/VQAv2_dataset.json"
VQA_original_image_path = "/work/images/VQAv2/VQAv2_train2014_2000"
VQA_image_path = "/work/images/VQAv2"




def load_ds(dataset_name, seed, add_options=None):

    user = os.environ['USER']

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
            'question': x['Question'], 'context': x['Body'], 'type': x['Type'],
            'equation': x['Equation'], 'id': x['ID'],
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

    elif dataset_name == "bioasq":
        # http://participants-area.bioasq.org/datasets/ we are using training 11b
        # could also download from here https://zenodo.org/records/7655130
        scratch_dir = os.getenv('SCRATCH_DIR', '.')
        path = f"{scratch_dir}/{user}/semantic_uncertainty/data/bioasq/training11b.json"
        with open(path, "rb") as file:
            data = json.load(file)

        questions = data["questions"]
        dataset_dict = {
            "question": [],
            "answers": [],
            "id": []
        }

        for question in questions:
            if "exact_answer" not in question:
                continue
            dataset_dict["question"].append(question["body"])
            if "exact_answer" in question:

                if isinstance(question['exact_answer'], list):
                    exact_answers = [
                        ans[0] if isinstance(ans, list) else ans
                        for ans in question['exact_answer']
                    ]
                else:
                    exact_answers = [question['exact_answer']]

                dataset_dict["answers"].append({
                    "text": exact_answers,
                    "answer_start": [0] * len(question["exact_answer"])
                })
            else:
                dataset_dict["answers"].append({
                    "text": question["ideal_answer"],
                    "answer_start": [0]
                })
            dataset_dict["id"].append(question["id"])

            dataset_dict["context"] = [None] * len(dataset_dict["id"])

        dataset = datasets.Dataset.from_dict(dataset_dict)

        # Split into training and validation set.
        dataset = dataset.train_test_split(test_size=0.8, seed=seed)
        train_dataset = dataset['train']
        validation_dataset = dataset['test']

    elif dataset_name == 'mmvp':

        def transformed_images_address(idx,base_path):
        # Iterate through folders
            image_paths = []
            # Iterate through folders
            for folder in os.listdir(base_path):
                folder_path = os.path.join(base_path, folder)
                if os.path.isdir(folder_path):
                    image_path = os.path.join(folder_path, f"{idx}.png")
                    if os.path.exists(image_path):

                        image_paths.append(image_path)

            # Print the total count
            return image_paths

        
        
        dataset_dict = datasets.load_dataset('csv', data_files=mmvp_dataset_path)
        dataset = dataset_dict['train']
        dataset = dataset.train_test_split(test_size=0.5, seed=42)
        train_dataset = dataset['train']
        validation_dataset = dataset['test']
        reformat = lambda x: {
            'id': x['Index'],
            'question': x['Question'],
            'options': x['Options'],
            'answers': {'text': x['Correct Answer']},
            'original_image':{'paths': os.path.join(mmvp_original_image_path, f"{x['Index']}.jpg")},
            'transformed_images': {'paths': transformed_images_address(x['Index'],mmvp_image_path)},
        }
        train_dataset = [reformat(d) for d in train_dataset]
        validation_dataset = [reformat(d) for d in validation_dataset]

    elif dataset_name == 'vqa':

        def transformed_images_address(idx,base_path):
        # Iterate through folders
            image_paths = []
            # Iterate through folders
            for folder in os.listdir(base_path):
                folder_path = os.path.join(base_path, folder)
                if os.path.isdir(folder_path):
                    image_path = os.path.join(folder_path, f"COCO_train2014_{str(idx).zfill(12)}.png")
                    if os.path.exists(image_path):

                        image_paths.append(image_path)

            # Print the total count
            return image_paths

        dataset_dict = datasets.load_dataset("json", data_files=VQA_dataset_path)
        dataset = dataset_dict['train'] 
        dataset = dataset.select(range(2000))
        dataset = dataset.train_test_split(test_size=0.5, seed=42)
        train_dataset = dataset['train']
        validation_dataset = dataset['test']

        reformat = lambda x: {
            'id': x['question_id'],
            'question': x['question'],
            'answers': {'text': x['answers']},
            'original_image':{'paths': os.path.join(VQA_original_image_path, f"COCO_train2014_{str(x['image_id']).zfill(12)}.jpg")},
            'transformed_images': {'paths': transformed_images_address(x['image_id'],VQA_image_path)},
        }

        print("Processing training dataset...")
        train_dataset = [reformat(d) for d in tqdm(train_dataset, desc="Reformatting Train Dataset")]

        print("Processing validation dataset...")
        validation_dataset = [reformat(d) for d in tqdm(validation_dataset, desc="Reformatting Validation Dataset")]




    else:
        raise ValueError

    return train_dataset, validation_dataset
