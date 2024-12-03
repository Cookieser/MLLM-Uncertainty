import gc
import os
import logging
import random
from tqdm import tqdm

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


def few_shot(args, train_dataset, prompt_indices):
    prompt = {}
    hint = "Please select the most correct option and only return the letter of the choice, such as (a) or (b):"
    prompt["hint"] = hint
    prompt["images"] = []  
    prompt["questions"] = []  
    prompt["answers"] = []  
    prompt['options'] = []  

    for idx in prompt_indices:
        image_path = train_dataset[idx]["original_image"]['paths']
        image = Image.open(image_path).convert("RGB")
        prompt["images"].append(image)
        prompt["questions"].append(train_dataset[idx]["question"])
        prompt["answers"].append(train_dataset[idx]["answers"]['text'])
        prompt['options'].append(train_dataset[idx]["options"])
    
    return prompt


def main(args):

    experiment_details = {'args': args}
    random.seed(args.random_seed)
    user = os.environ['USER']
    slurm_jobid = os.getenv('SLURM_JOB_ID', None)
    scratch_dir = os.getenv('SCRATCH_DIR', '.')
    if not os.path.exists(f"{scratch_dir}/{user}/uncertainty"):
        os.makedirs(f"{scratch_dir}/{user}/uncertainty")

    wandb.init(
        entity=args.entity,
        project="semantic_uncertainty" if not args.debug else "semantic_uncertainty_debug",
        dir=f"{scratch_dir}/{user}/uncertainty",
        config=args,
        notes=f'slurm_id: {slurm_jobid}, experiment_lot: {args.experiment_lot}',
    )
    logging.info('Finished wandb init.')

    # Get accuracy metric.
    metric = utils.get_metric(args.metric)
    # Load dataset.
    train_dataset,validation_dataset,answerable_indices,unanswerable_indices = dataset_handler(args)

    prompt_indices = random.sample(answerable_indices, args.num_few_shot)
    experiment_details['prompt_indices'] = prompt_indices

    remaining_answerable = list(set(answerable_indices) - set(prompt_indices))

    few_shot_info = few_shot(args, train_dataset, prompt_indices)



    model = MLLMModel('llava-1.5-7b-hf','default',50);

    q = train_dataset[0]["question"]
    i = train_dataset[0]["original_image"]["paths"]
    options = train_dataset[0]["options"]
    image = Image.open(i).convert("RGB")
    #answer,clean_log_likelihoods,last_token_embedding = model.predict_few_shot(q,image,few_shot_info,1, return_full=False)
    answer,clean_log_likelihoods,last_token_embedding= model.predict_few_shot(q,image,options,few_shot_info,1, return_full=False)


    print("Clean Answer:", answer)
    print("Log-Likelihoods:", clean_log_likelihoods)
    print("Last Token Embedding Shape:", last_token_embedding.shape)

    # Initialize prompt for p_true baseline.
    if args.compute_p_true:
        logging.info(80*'#')
        logging.info('Constructing few-shot prompt for p_true.')

        p_true_indices = random.sample(answerable_indices, args.p_true_num_fewshot)
        remaining_answerable = list(set(remaining_answerable) - set(p_true_indices))

        #p_true_few_shot_prompt, p_true_responses, len_p_true = p_true_handler(args,model,train_dataset,p_true_indices,prompt,BRIEF,make_prompt,metric)
        
        # wandb.config.update(
        #     {'p_true_num_fewshot': len_p_true}, allow_val_change=True)
        # wandb.log(dict(len_p_true=len_p_true))
        # experiment_details['p_true_indices'] = p_true_indices
        # experiment_details['p_true_responses'] = p_true_responses
        # experiment_details['p_true_few_shot_prompt'] = p_true_few_shot_prompt
        # logging.info('Finished constructing few-shot prompt for p_true.')
        # logging.info(80*'#')
        # logging.info('p_true_few_shot_prompt: %s', p_true_few_shot_prompt)
        # logging.info(80*'#')


if __name__ == '__main__':

    parser = utils.get_parser()
    args, unknown = parser.parse_known_args()
    logging.info('Starting new run with args: %s', args)

    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    if args.compute_uncertainties:
        args.assign_new_wandb_id = False

    # First sample generations from LLM.
    logging.info('STARTING `generate_answers`!')
    main(args)
    logging.info('FINISHED `generate_answers`!')













