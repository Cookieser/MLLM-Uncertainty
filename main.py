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


def p_true(model,dataset,indices,prompt_info,num_generations,metric):
    conversation =[]
    images = []
    all_responses = dict()
    for it, i in enumerate(indices):
        prompt_candidate = []
        example = dataset[i]
        question = example["question"]
        image_file = example["original_image"]['paths']
        image = Image.open(image_file).convert("RGB")
        images.append(image)
        options = example["options"]


        if it != 0:
            prompt_candidate += ['\n']
        prompt_candidate += ['Question: ' + question]
        prompt_candidate += ['\nBrainstormed Answers: ']
        logging.info('P_TRUE >> Current Question: '.ljust(25) + question)

        responses = []
        for j in range(num_generations + 1):
            if j == 0:
                temperature = 0.1
            else:
                temperature = 1.0
            response, _, _ = model.predict_few_shot(question,image,options,prompt_info,temperature)
            logging.info('P_TRUE >> Current Response: '.ljust(25) + response)
            responses.append(response)
            prompt_candidate += [f'{response.strip()} \n']
            if j == 0:
                # Save most likely response and compute correctness metric for it.
                most_likely_response = response
                is_correct = True  ################
                answers = [answer for answer in example['answers']['text']] #############
                logging.info('P_TRUE >> LOW-T >> true answer: '.ljust(35) + str(answers))
                logging.info('P_TRUE >> LOW-T >> acc: '.ljust(35) + str(is_correct))
        all_responses[i] = dict(
            responses=responses, most_likely_response=most_likely_response,
            is_correct=is_correct)

        prompt_candidate += ['Possible answer: ' + most_likely_response + '\n']
        prompt_candidate += ['Is the possible answer:\n']
        prompt_candidate += ['A) True\n']
        prompt_candidate += ['B) False\n']
        prompt_candidate += ['The possible answer is:']
        mllm_answer = 'A' if is_correct else 'B'
        conversation.append( 
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_candidate},
                    {"type": "image"},  
                    
                ],
            })
        conversation.append(
        {
        "role": "assistant",
        "content": [
        {"type": "text", "text": mllm_answer},
            ],
        })

    return conversation,images,all_responses




def generation(args,dataset,dataset_split,indices,prompt_info,model,metric,accuracies, generations,  p_trues):
    it = 0
    for index in tqdm(indices):
        if (it + 1 % 10) == 0:
            gc.collect()
            torch.cuda.empty_cache()
        it += 1

        # Grab example at index.
        example = dataset[index]
        

        example = dataset[index]
        question = example["question"]
        image_file = example["original_image"]['paths']
        image = Image.open(image_file).convert("RGB")
        options = example["options"]
        correct_answer =example['answers']['text']

        generations[example['id']] = {'question': question, 'image_path': image_file}


        logging.info('Current input: '.ljust(15) + question)

        full_responses = []


        # We sample one low temperature answer on which we will compute the
        # accuracy and args.num_generation high temperature answers which will
        # be used to estimate the entropy variants.

        if dataset_split == 'train' and args.get_training_set_generations_most_likely_only:
            num_generations = 1
        else:
            num_generations = args.num_generations + 1

        for i in range(num_generations):

            # Temperature for first generation is always `0.1`.
            temperature = 0.1 if i == 0 else args.temperature

            predicted_answer, token_log_likelihoods, embedding = model.predict_few_shot(question,image,options,prompt_info,temperature)
            embedding = embedding.cpu() if embedding is not None else None

            # Only compute accuracy if question is answerable.
            compute_acc = args.compute_accuracy_at_all_temps or (i == 0)
            if correct_answer and compute_acc:
                #acc = metric(predicted_answer, example, model)
                acc = 1.0
            else:
                acc = 0.0  # pylint: disable=invalid-name
            


            if i == 0:
                logging.info('Iteration ' + str(it) + ':  ' + 80*'#')

                logging.info('question: '.ljust(15) + question)
                logging.info('image: '.ljust(15) + image_file)
                logging.info('low-t prediction: '.ljust(15) + predicted_answer)
                logging.info('correct answer: '.ljust(15) + str(correct_answer))
                logging.info('accuracy: '.ljust(15) + str(acc))

                accuracies.append(acc)
                most_likely_answer_dict = {
                    'response': predicted_answer,
                    'token_log_likelihoods': token_log_likelihoods,
                    'embedding': embedding,
                    'accuracy': acc}
                generations[example['id']].update({
                    'most_likely_answer': most_likely_answer_dict,
                    'reference': utils.get_reference(example)})
                
            else:
                logging.info('high-t prediction '.ljust(15) + str(i) + ' : ' + predicted_answer)
                # Aggregate predictions over num_generations.
                full_responses.append(
                    (predicted_answer, token_log_likelihoods, embedding, acc))
        

        # Append all predictions for this example to `generations`.
        generations[example['id']]['responses'] = full_responses

        #if args.compute_p_true and dataset_split == 'validation':
            # Already compute p_true here. Avoid cost of generations in compute_uncertainty script.
            #p_true = calculate_mllm_p_true
            #p_trues.append(p_true)
            #logging.info('p_true: %s', p_true)
    return generations,accuracies 





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


    # Initialize prompt for p_true baseline.
    if args.compute_p_true:
        logging.info(80*'#')
        logging.info('Constructing few-shot prompt for p_true.')

        p_true_indices = random.sample(answerable_indices, args.p_true_num_fewshot)
        experiment_details['p_true_indices'] = p_true_indices
        remaining_answerable = list(set(remaining_answerable) - set(p_true_indices))

        conversation,images,p_true_responses = p_true(model,train_dataset,p_true_indices,few_shot_info,args.num_generations,metric)

        experiment_details['p_true_responses'] = p_true_responses

        experiment_details['conversation'] = conversation
        logging.info('conversation: %s', conversation)

        logging.info('Finished constructing few-shot prompt for p_true.')
        logging.info(80*'#')
        
        logging.info(80*'#')


        # Start answer generation.
    logging.info(80 * '=')
    logging.info('Generating answers: ')
    logging.info(80 * '=')
    for dataset_split in ['train', 'validation']:
        logging.info(80 * 'x')
        logging.info('Starting with dataset_split %s.', dataset_split)
        logging.info(80 * 'x')

        # This will store all input data and model predictions.
        accuracies, generations, results_dict, p_trues = [], {}, {}, []
        if dataset_split == 'train':
            if not args.get_training_set_generations:
                logging.info('Skip training data.')
                continue
            dataset = train_dataset
            possible_indices = list(set(remaining_answerable) | set(unanswerable_indices))       

        else:
            dataset = validation_dataset
            possible_indices = range(0, len(dataset))

        # Evaluate over random subset of the datasets.
        indices = random.sample(possible_indices, min(args.num_samples, len(dataset)))
        experiment_details[dataset_split] = {'indices': indices}

        if args.num_samples > len(dataset):
            logging.warning('Not enough samples in dataset. Using all %d samples.', len(dataset))

        #generations,accuracies,p_trues = generation_handler(args,dataset,dataset_split,indices,make_prompt,BRIEF,prompt,p_true_few_shot_prompt,model,metric,accuracies, generations, p_trues)
        generations,accuracies = generation(args,dataset,dataset_split,indices,few_shot_info,model,metric,accuracies, generations,  p_trues)
        # Save generations for that split.
        utils.save(generations, f'{dataset_split}_generations.pkl')

        # Log overall accuracy.
        accuracy = np.mean(accuracies)
        print(f"Overall {dataset_split} split accuracy: {accuracy}")
        wandb.log({f"{dataset_split}_accuracy": accuracy})


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













