import yaml
import os
import math
from src.dataset import Dataset
from src.prompt_engineer import PromptGenerator
from src.models import HuggingfaceModel
from src.samplers import Sampler
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

with open("configs/experiment_config1.yaml", "r") as file:
    config = yaml.safe_load(file)

dataset_loader = Dataset(config)
train_dataset, validation_dataset = dataset_loader.load_data()



promptgenerator = PromptGenerator(config,validation_dataset)
prompt = promptgenerator.generate_prompt_by_id(1)
prompts =promptgenerator.generate_prompts_by_count(5)



huggingface_model = HuggingfaceModel(config)


sampler = Sampler(prompt,huggingface_model,sample_method="simple")
sampler.sample(8)
sampler.show_all_result()






