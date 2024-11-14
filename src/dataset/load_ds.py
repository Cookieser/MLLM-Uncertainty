import hashlib
import datasets
class Dataset:
    def __init__(self, config):
        self.dataset_name = config['experiment']['dataset']
        self.seed = config['experiment']['seed']

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



