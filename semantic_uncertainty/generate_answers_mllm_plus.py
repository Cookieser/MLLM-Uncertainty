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
from uncertainty.accuracy_metric.metrics import *

utils.setup_logger()


def dataset_handler(args):
    # Setup run.
    if args.dataset == 'svamp':
        if not args.use_context:
            logging.info('Forcing `use_context=True` for svamp dataset.')
            args.use_context = True
    elif args.dataset == 'squad':
        if not args.answerable_only:
            logging.info('Forcing `answerable_only=True` for squad dataset.')
            args.answerable_only = True

    # Load dataset.
    train_dataset, validation_dataset = load_ds(
        args.dataset, add_options=args.use_mc_options, seed=args.random_seed)
    if args.ood_train_dataset is not None:
        logging.warning(
            'Using OOD dataset %s to construct few-shot prompts and train p_ik.',
            args.ood_train_dataset)
        # Get indices of answerable and unanswerable questions and construct prompt.
        train_dataset, _ = load_ds(args.ood_train_dataset, add_options=args.use_mc_options)
    if not isinstance(train_dataset, list):
        logging.info('Train dataset: %s', train_dataset)

    # Get indices of answerable and unanswerable questions and construct prompt.
    answerable_indices, unanswerable_indices = utils.split_dataset(train_dataset)

    if args.answerable_only:
        unanswerable_indices = []
        val_answerable, val_unanswerable = utils.split_dataset(validation_dataset)
        del val_unanswerable
        validation_dataset = [validation_dataset[i] for i in val_answerable]
    return train_dataset,validation_dataset,answerable_indices,unanswerable_indices

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

        transformed_image_path_list = example['transformed_images']['paths']
        nums_len_transformed_image = len(transformed_image_path_list)


        answer = example['answers']['text']

        if isinstance(answer, list):
                answer = ", ".join(answer)  
        elif not isinstance(answer, str):
            raise ValueError(f"answer must be a string or list of strings, but got {type(answer)}")


        if it != 0:
            prompt_candidate += ['\n']
        prompt_candidate += ['Question: ' + question]
        prompt_candidate += ['\nBrainstormed Answers: ']
        logging.info('P_TRUE >> Current Question: '.ljust(25) + question)

        responses = []
        for j in range(num_generations + 1):
            temperature = 0.1
            if j == 0:
                image = image
            else:
                if(j > nums_len_transformed_image):
                    raise ValueError("Index exceeds the length of transformed image array.")
                
                
                transformed_image_path = example['transformed_images']['paths'][j-1]
                transformed_image = Image.open(transformed_image_path).convert("RGB")
                image = transformed_image

            response, _, _ = model.predict_few_shot(question,image,options,prompt_info,temperature)
            logging.info('P_TRUE >> Current Response: '.ljust(25) + response)
            responses.append(response)
            prompt_candidate += [f'{response.strip()} \n']
            if j == 0:
                # Save most likely response and compute correctness metric for it.
                most_likely_response = response
                is_correct = metric(most_likely_response, example)  ################
                logging.info('P_TRUE >> Original >> true answer: '.ljust(35) + str(answer))
                logging.info('P_TRUE >> Original >> acc: '.ljust(35) + str(is_correct))
        all_responses[i] = dict(
            responses=responses, most_likely_response=most_likely_response,
            is_correct=is_correct)

        prompt_candidate += ['Possible answer: ' + most_likely_response + '\n']
        prompt_candidate += ['Is the possible answer:\n']
        prompt_candidate += ['A) True\n']
        prompt_candidate += ['B) False\n']
        prompt_candidate += ['The possible answer is:']
        prompt_candidate_str = ''.join(prompt_candidate)
        mllm_answer = 'A' if is_correct else 'B'
        conversation.append( 
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_candidate_str},
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



def calculate_mllm_p_true(model,question,image,most_probable_answer, brainstormed_answers,few_shot_conversation,few_shot_images):
    conversation = copy.deepcopy(few_shot_conversation)
    few_shot_images= copy.deepcopy(few_shot_images)
    prompt = ''
    prompt += 'Question: ' + question
    prompt += '\nBrainstormed Answers: '
    for answer in brainstormed_answers + [most_probable_answer]:
        prompt += answer.strip() + '\n'
    prompt += 'Possible answer: ' + most_probable_answer + '\n'
    prompt += 'Is the possible answer:\n'
    prompt += 'A) True\n'
    prompt += 'B) False\n'
    prompt += 'The possible answer is:'
    prompt += 'Do the brainstormed answers match the possible answer? Respond with A if they do, if they do not respond with B. Answer:'

    conversation.append( 
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},  
                    
                ],
            })

    few_shot_images.append(image)
    log_prob = model.get_p_true(conversation,few_shot_images)

    return log_prob

def generation(args,dataset,dataset_split,indices,prompt_info,model,metric,accuracies, generations,  p_trues,few_shot_conversation,few_shot_images):
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
        transformed_image_path_list = example['transformed_images']['paths']
        nums_len_transformed_image = len(transformed_image_path_list)
        options = example["options"]
        correct_answer =example['answers']['text']
        if isinstance(correct_answer, list):
                correct_answer = ", ".join(correct_answer)  
        elif not isinstance(correct_answer, str):
            raise ValueError(f"answer must be a string or list of strings, but got {type(correct_answer)}")


        generations[example['id']] = {'question': question, 'image_path': image_file}


        logging.info('Current input: '.ljust(15) + question)

        full_responses = []




        if dataset_split == 'train' and args.get_training_set_generations_most_likely_only:
            num_generations = 1
        else:
            num_generations = args.num_generations + 1

        for i in range(num_generations):
            temperature = 0.1

            if(i == 0):
                #print(f"We use the original image")
                image = image
            else:
                if(i > nums_len_transformed_image):
                    raise ValueError("Index exceeds the length of transformed image array.")
                #print(f"We use the {i}_transformed_image")
                transformed_image_path = example['transformed_images']['paths'][i-1]
                transformed_image = Image.open(transformed_image_path).convert("RGB")
                image = transformed_image
                image_file = transformed_image_path


            predicted_answer, token_log_likelihoods, embedding = model.predict_few_shot(question,image,options,prompt_info,temperature)
            embedding = embedding.cpu() if embedding is not None else None

            # Only compute accuracy if question is answerable.
            compute_acc = args.compute_accuracy_at_all_temps or (i == 0)
            if correct_answer and compute_acc:
                acc = metric(predicted_answer, example)
            else:
                acc = 0.0  # pylint: disable=invalid-name
            


            if i == 0:
                logging.info('Iteration ' + str(it) + ':  ' + 80*'#')

                logging.info('question: '.ljust(15) + question)
                logging.info('original image: '.ljust(15) + image_file)
                logging.info('original prediction: '.ljust(15) + predicted_answer)
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
                    'reference': get_reference(example)})
                
            else:
                logging.info('transformed image: '.ljust(15) + image_file)
                logging.info('transformed prediction '.ljust(15) + str(i) + ' : ' + predicted_answer)
                # Aggregate predictions over num_generations.
                full_responses.append(
                    (predicted_answer, token_log_likelihoods, embedding, acc))
        

        # Append all predictions for this example to `generations`.
        generations[example['id']]['responses'] = full_responses

        if args.compute_p_true and dataset_split == 'validation':
            # Already compute p_true here. Avoid cost of generations in compute_uncertainty script.
            p_true = calculate_mllm_p_true(model,question,image,most_likely_answer_dict['response'], [r[0] for r in full_responses],few_shot_conversation,few_shot_images)
            p_trues.append(p_true)
            logging.info('p_true: %s', p_true)
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
    metric = get_metric('mc')
    # Load dataset.
    train_dataset,validation_dataset,answerable_indices,unanswerable_indices = dataset_handler(args)


    # Create Few-Shot prompt.
    prompt_indices = random.sample(answerable_indices, args.num_few_shot)
    experiment_details['prompt_indices'] = prompt_indices

    remaining_answerable = list(set(answerable_indices) - set(prompt_indices))

    few_shot_info = few_shot(args, train_dataset, prompt_indices)


    # Initialize model.
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

        generations,accuracies = generation(args,dataset,dataset_split,indices,few_shot_info,model,metric,accuracies, generations,  p_trues,conversation,images)
        # Save generations for that split.
        utils.save(generations, f'{dataset_split}_generations.pkl')


        # Log overall accuracy.
        accuracy = np.mean(accuracies)
        print(f"Overall {dataset_split} split accuracy: {accuracy}")
        wandb.log({f"{dataset_split}_accuracy": accuracy})

        if dataset_split == 'validation':
            if args.compute_p_true:
                results_dict['uncertainty_measures'] = {
                    'p_false':  [1 - p for p in p_trues],
                    'p_false_fixed':  [1 - np.exp(p) for p in p_trues],
                }
            utils.save(results_dict, 'uncertainty_measures.pkl')

    utils.save(experiment_details, 'experiment_details.pkl')
    logging.info('Run complete.')
    del model


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

    if args.compute_uncertainties:
        # Follow with uncertainty calculation script by default.
        args.assign_new_wandb_id = False
        gc.collect()
        torch.cuda.empty_cache()
        logging.info(50 * '#X')
        logging.info('STARTING `compute_uncertainty_measures`!')
        main_compute(args)
        logging.info('FINISHED `compute_uncertainty_measures`!')













