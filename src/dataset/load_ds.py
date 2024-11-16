import hashlib
import datasets
class Dataset:
    def __init__(self, config):
        self.dataset_name = config['dataset']['name']
        self.seed = config['dataset']['seed']

        self.loaders = {
            "squad": self._load_squad,
            "svamp": self._load_svamp,
            "nq": self._load_nq,
            "trivia_qa": self._load_trivia_qa,
            "truthful_mc1": self._load_truthful_qa_mc1,
            "trustful_mc2": self._load_truthful_qa_mc2,
        }
    def load_data(self):
        if self.dataset_name in self.loaders:
            return self.loaders[self.dataset_name]()
        else:
            raise ValueError(f"Dataset '{self.dataset_name}' is not recognized.")   

# Individual dataset preprocessing functions
    def _load_squad(self):
        dataset = datasets.load_dataset("squad_v2")
        return dataset["train"], dataset["validation"]

    def _load_svamp(self):
        dataset = datasets.load_dataset('ChilleD/SVAMP')
        reformat = lambda x: {
            'question': x['Question'], 
            'context': x['Body'], 
            'type': x['Type'],
            'equation': x['Equation'], 
            'id': x['ID'],
            'answers': {'text': [str(x['Answer'])]}
        }
        train_dataset = [reformat(d) for d in dataset["train"]]
        validation_dataset = [reformat(d) for d in dataset["test"]]
        return train_dataset, validation_dataset

    def _load_nq(self):
        dataset = datasets.load_dataset("nq_open")
        md5hash = lambda s: str(int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16))
        reformat = lambda x: {
            'question': x['question'] + '?',
            'answers': {'text': x['answer']},
            'context': '',
            'id': md5hash(str(x['question'])),
        }
        train_dataset = [reformat(d) for d in dataset["train"]]
        validation_dataset = [reformat(d) for d in dataset["validation"]]
        return train_dataset, validation_dataset

    def _load_trivia_qa(self,seed):
        dataset = datasets.load_dataset('TimoImhof/TriviaQA-in-SQuAD-format')['unmodified']
        dataset = dataset.train_test_split(test_size=0.2, seed=seed)
        return dataset['train'], dataset['test']

    def _load_truthful_qa_mc1(self):
        dataset = datasets.load_dataset("truthfulqa/truthful_qa", 'multiple_choice')
        md5hash = lambda s: str(int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16))
        reformat = lambda x: {
            'question': x['question'],
            'choices': x['mc1_targets']['choices'],
            'answers': {'text': x['mc1_targets']['labels']},
            'id': md5hash(str(x['question'])),
        }
        validation_dataset = [reformat(d) for d in dataset["validation"]]
        return None, validation_dataset  # No training set for this dataset

    def _load_truthful_qa_mc2(self):
        dataset = datasets.load_dataset("truthfulqa/truthful_qa", 'multiple_choice')
        md5hash = lambda s: str(int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16))
        reformat = lambda x: {
            'question': x['question'],
            'choices': x['mc2_targets']['choices'],
            'answers': {'text': x['mc2_targets']['labels']},
            'id': md5hash(str(x['question'])),
        }
        validation_dataset = [reformat(d) for d in dataset["validation"]]
        return None, validation_dataset  # No training set for this dataset


    # def _load_bioasq(self):
    #     # http://participants-area.bioasq.org/datasets/ we are using training 11b
    #     # could also download from here https://zenodo.org/records/7655130
    #     scratch_dir = os.getenv('SCRATCH_DIR', '.')
    #     path = f"{scratch_dir}/{user}/semantic_uncertainty/data/bioasq/training11b.json"
    #     with open(path, "rb") as file:
    #         data = json.load(file)

    #     questions = data["questions"]
    #     dataset_dict = {
    #         "question": [],
    #         "answers": [],
    #         "id": []
    #     }

    #     for question in questions:
    #         if "exact_answer" not in question:
    #             continue
    #         dataset_dict["question"].append(question["body"])
    #         if "exact_answer" in question:

    #             if isinstance(question['exact_answer'], list):
    #                 exact_answers = [
    #                     ans[0] if isinstance(ans, list) else ans
    #                     for ans in question['exact_answer']
    #                 ]
    #             else:
    #                 exact_answers = [question['exact_answer']]

    #             dataset_dict["answers"].append({
    #                 "text": exact_answers,
    #                 "answer_start": [0] * len(question["exact_answer"])
    #             })
    #         else:
    #             dataset_dict["answers"].append({
    #                 "text": question["ideal_answer"],
    #                 "answer_start": [0]
    #             })
    #         dataset_dict["id"].append(question["id"])

    #         dataset_dict["context"] = [None] * len(dataset_dict["id"])

    #     dataset = datasets.Dataset.from_dict(dataset_dict)

    #     # Split into training and validation set.
    #     dataset = dataset.train_test_split(test_size=0.8, seed=seed)
    #     train_dataset = dataset['train']
    #     validation_dataset = dataset['test']
    #     return train_dataset,validation_dataset



