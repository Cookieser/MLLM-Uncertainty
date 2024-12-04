import gc
import os
import logging
import random
from tqdm import tqdm
import copy
import numpy as np
import torch
import wandb

from PIL import Image
from uncertainty.models.mllm_model import MLLMModel
from uncertainty.models.huggingface_models import HuggingfaceModel

from uncertainty.data.data_utils import load_ds
from uncertainty.utils import utils
from uncertainty.uncertainty_measures import p_true as p_true_utils
from compute_uncertainty_measures import main as main_compute
from dataset_handle import dataset_handler
from few_shot_handle import few_shot_handler
from p_true_handle import p_true_handler
from generation_handle import generation_handler
utils.setup_logger()

model = MLLMModel('llava-1.5-7b-hf','default',50);


#few_shot_conversation = [{'role': 'user', 'content': [{'type': 'text', 'text': "Question: Is the butterfly's abdomen visible in the image?\nBrainstormed Answers: (a) \n(a) \n(a) \n(b) \n(a) \n(a) \n(a) \n(a) \n(a) \n(a) \n(a) \nPossible answer: (a)\nIs the possible answer:\nA) True\nB) False\nThe possible answer is:"}, {'type': 'image'}]}, {'role': 'assistant', 'content': [{'type': 'text', 'text': 'A'}]}, {'role': 'user', 'content': [{'type': 'text', 'text': '\nQuestion: Is the shoe in the image tied or untied?\nBrainstormed Answers: (a) \n(b) \n(a) \n(a) \n(b) \n(a) \n(a) \n(a) \n(a) \n(a) \n(a) \nPossible answer: (a)\nIs the possible answer:\nA) True\nB) False\nThe possible answer is:'}, {'type': 'image'}]}, {'role': 'assistant', 'content': [{'type': 'text', 'text': 'B'}]}, {'role': 'user', 'content': [{'type': 'text', 'text': '\nQuestion: Is the minute hand of the clock closer to 12 or closer to 1?\nBrainstormed Answers: (b) \n(a) \n(a) \n(b) \n(a) \n(b) \n(a) \n(a) \n(a) \n(a) \n(a) \nPossible answer: (b)\nIs the possible answer:\nA) True\nB) False\nThe possible answer is:'}, {'type': 'image'}]}, {'role': 'assistant', 'content': [{'type': 'text', 'text': 'B'}]}, {'role': 'user', 'content': [{'type': 'text', 'text': 'Question: Can you see the dorsal fin of the animal?\nBrainstormed Answers: (a)\n(a)\n(a)\n(b)\nNo\n(a)\n(a)\n(b)\n(a)\n(a)\n(a)\nPossible answer: (a)\nIs the possible answer:\nA) True\nB) False\nThe possible answer is:Do the brainstormed answers match the possible answer? Respond with A if they do, if they do not respond with B. Answer:'}, {'type': 'image'}]}, {'role': 'assistant', 'content': [{'type': 'text', 'text': 'A'}]}]


conversation =[{'role': 'user', 'content': [{'type': 'text', 'text': "Question: Is the butterfly's abdomen visible in the image?\nBrainstormed Answers: (a) \n(a) \n(a) \n(b) \n(a) \n(a) \n(a) \n(a) \n(a) \n(a) \n(a) \nPossible answer: (a)\nIs the possible answer:\nA) True\nB) False\nThe possible answer is:"}, {'type': 'image'}]}]
#conversation.append()

image_path = "/work/images/images/mmvp/1.jpg"
images =[]
image = Image.open(image_path).convert('RGB')
images.append(image)
text ="Question: Is the butterfly's abdomen visible in the image?\nBrainstormed Answers: (a) \n(a) \n(a) \n(b) \n(a) \n(a) \n(a) \n(a) \n(a) \n(a) \n(a) \nPossible answer: (a)\nIs the possible answer:\nA) True\nB) False\nThe possible answer is:"


p = model.get_p_true(conversation,image)

print(f"p_true:{p}")