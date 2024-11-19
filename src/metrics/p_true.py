#  Language Models (Mostly) Know What They Know

#  Saurav Kadavath et al.,21 Nov 2022 
#  <https://arxiv.org/abs/2207.05221>

import random
import logging
from evaluate import load
from tqdm import tqdm 
import wandb
import pickle
import numpy as np


class PTrueEvaluator:
    def __init__(self, config,model,train_promptgenerator, validation_promptgenerator,metric,experiment_details):
        p_ture_config = config["p_true"]
        self.get_training_set_generations= p_ture_config["get_training_set_generations"]
        self.get_training_set_generations_most_likely_only = p_ture_config["get_training_set_generations_most_likely_only"]
        self.compute_accuracy_at_all_temps = p_ture_config["compute_accuracy_at_all_temps"]
        self.compute_p_true = p_ture_config["compute_p_true"]
        self.p_true_hint = p_ture_config["p_true_hint"]
        self.train_promptgenerator = train_promptgenerator
        self.validation_promptgenerator = validation_promptgenerator
        self.model =model
        self.metric = metric
        self.experiment_details = experiment_details



        

    def all_evaluate(self,prompt,temperature,num_generate_answers,p_true_few_shot_prompt,num_samples):
        logging.info(80 * 'x')
        logging.info('Starting with dataset_split %s.', "train")
        logging.info(80 * 'x')
        if not self.get_training_set_generations:
            logging.info('Skip training data.')
        else:
            self.train_evaluate(prompt,temperature,num_generate_answers,num_samples)
        
        self.validation_evaluate(prompt,temperature,num_generate_answers,p_true_few_shot_prompt,num_samples)

        self.save_wandb(self.experiment_details, 'experiment_details.pkl')
        logging.info('Run complete.')




    def train_evaluate(self,prompt,temperature,num_generate_answers,num_samples):
        unused_indices = self.train_promptgenerator.get_unused_indices()
        logging.info('Unused items in train dataset: %d', len(unused_indices))
        if num_samples > len(unused_indices):
            logging.warning('Not enough samples in dataset. Using all %d samples.', len(unused_indices))
        indices = random.sample(unused_indices,min(num_samples, len(unused_indices)))
        self.experiment_details['train'] = {'indices': indices}
        accuracies, generations, results_dict, p_trues = [], {}, {}, []
        for it, index in enumerate(tqdm(indices)):  
            if (it + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            example = self.train_promptgenerator.dataset_item[index]
            question, context = example["question"], example['context']
            generations[example['id']] = {'question': question, 'context': context}
            correct_answer = example['answers']['text']
            current_input = self.train_promptgenerator._make_prompt(context, question, None, self.train_promptgenerator.BRIEF)
            local_prompt = prompt + current_input

            full_responses = []

            if self.get_training_set_generations_most_likely_only:
                num_generations = 1
            else:
                num_generations = num_generate_answers + 1
            
            for i in range(num_generations):
                # Temperature for first generation is always `0.1`.
                temperature = 0.1 if i == 0 else temperature
                predicted_answer, token_log_likelihoods = self.model.predict(local_prompt, temperature)
                compute_acc = self.compute_accuracy_at_all_temps or (i == 0)
                if correct_answer and compute_acc:
                    acc = self.metric(predicted_answer, example, self.model)
                else:
                    acc = 0.0  
                if i == 0:
                    logging.info('Iteration ' + str(it) + ':  ' + 80*'#')

                    logging.info('Current input: '.ljust(15) + current_input)
                    if self.train_promptgenerator.use_contexts:
                        logging.info('context: '.ljust(15) + str(context))
                    logging.info('question: '.ljust(15) + question)
                    logging.info('low-t prediction: '.ljust(15) + predicted_answer)
                    logging.info('correct answer: '.ljust(15) + str(correct_answer))
                    logging.info('accuracy: '.ljust(15) + str(acc))

                    accuracies.append(acc)
                    most_likely_answer_dict = {
                        'response': predicted_answer,
                        'token_log_likelihoods': token_log_likelihoods,
                        #'embedding': embedding,
                        'accuracy': acc}
                    generations[example['id']].update({
                        'most_likely_answer': most_likely_answer_dict,
                        'reference': self.get_reference(example)})
                else:
                    logging.info('high-t prediction '.ljust(15) + str(i) + ' : ' + predicted_answer)
                    # Aggregate predictions over num_generations.
                    full_responses.append((predicted_answer, token_log_likelihoods, acc))
            
            generations[example['id']]['responses'] = full_responses
        self.save_wandb(generations, 'train_generations.pkl')
        accuracy = np.mean(accuracies)
        logging.info(f"Overall train split accuracy: {accuracy}")

                



    def validation_evaluate(self,prompt,temperature,num_generate_answers,p_true_few_shot_prompt,num_samples):
        unused_indices = self.validation_promptgenerator.get_unused_indices()
        logging.info('Unused items in train dataset: %d', len(unused_indices))
        if num_samples > len(unused_indices):
            logging.warning('Not enough samples in dataset. Using all %d samples.', len(unused_indices))
        indices = random.sample(unused_indices,min(num_samples, len(unused_indices)))
        self.experiment_details['validation'] = {'indices': indices}
        accuracies, generations, results_dict, p_trues = [], {}, {}, []
        for it, index in enumerate(tqdm(indices)):  
            if (it + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            example = self.validation_promptgenerator.dataset_item[index]
            question, context = example["question"], example['context']
            generations[example['id']] = {'question': question, 'context': context}
            correct_answer = example['answers']['text']
            current_input = self.validation_promptgenerator._make_prompt(context, question, None, self.validation_promptgenerator.BRIEF)
            local_prompt = prompt + current_input

            full_responses = []

            
            num_generations = num_generate_answers + 1
            
            for i in range(num_generations):
                # Temperature for first generation is always `0.1`.
                temperature = 0.1 if i == 0 else temperature
                predicted_answer, token_log_likelihoods = self.model.predict(local_prompt, temperature)
                compute_acc = self.compute_accuracy_at_all_temps or (i == 0)
                if correct_answer and compute_acc:
                    acc = self.metric(predicted_answer, example, self.model)
                else:
                    acc = 0.0  
                if i == 0:
                    logging.info('Iteration ' + str(it) + ':  ' + 80*'#')

                    logging.info('Current input: '.ljust(15) + current_input)
                    if self.validation_promptgenerator.use_contexts:
                        logging.info('context: '.ljust(15) + str(context))
                    logging.info('question: '.ljust(15) + question)
                    logging.info('low-t prediction: '.ljust(15) + predicted_answer)
                    logging.info('correct answer: '.ljust(15) + str(correct_answer))
                    logging.info('accuracy: '.ljust(15) + str(acc))

                    accuracies.append(acc)
                    most_likely_answer_dict = {
                        'response': predicted_answer,
                        'token_log_likelihoods': token_log_likelihoods,
                        #'embedding': embedding,
                        'accuracy': acc}
                    generations[example['id']].update({
                        'most_likely_answer': most_likely_answer_dict,
                        'reference': self.get_reference(example)})
                else:
                    logging.info('high-t prediction '.ljust(15) + str(i) + ' : ' + predicted_answer)
                    # Aggregate predictions over num_generations.
                    full_responses.append((predicted_answer, token_log_likelihoods, acc))
            
            generations[example['id']]['responses'] = full_responses

            if self.compute_p_true:
                # Already compute p_true here. Avoid cost of generations in compute_uncertainty script.
                p_true = self.calculate_p_true(
                    self.model, question, most_likely_answer_dict['response'],
                    [r[0] for r in full_responses], p_true_few_shot_prompt,
                    hint=self.p_true_hint)
                p_trues.append(p_true)
                logging.info('p_true: %s', p_true)

        self.save_wandb(generations, 'validation_generations.pkl')

        accuracy = np.mean(accuracies)
        logging.info(f"Overall validation split accuracy: {accuracy}")

        if self.compute_p_true:
            results_dict['uncertainty_measures'] = {
                'p_false':  [1 - p for p in p_trues],
                'p_false_fixed':  [1 - np.exp(p) for p in p_trues],
            }
        self.save_wandb(results_dict, 'uncertainty_measures.pkl')




    def construct_few_shot_prompt_for_p_true(self, prompt,num_generations,p_true_num_fewshot):
        """Construct few shot prompt for p_true uncertainty metric."""
        p_true_indices = random.sample(list(self.train_promptgenerator.unused_indices), p_true_num_fewshot)

        # Validate that all requested indices are in the unused set
        if not set(p_true_indices).issubset(self.train_promptgenerator.unused_indices):
                raise ValueError("Some of the provided indices are not available in the unused set.")

        # Call model n_shots many times.
        few_shot_prompt = []
        all_responses = dict()


        for it, i in enumerate(p_true_indices):
            prompt_candidate = []
            example = self.train_promptgenerator.dataset_item[i]
            question = example["question"]
            context = example["context"]
            if it != 0:
                prompt_candidate += ['\n']
            prompt_candidate += ['Question: ' + question]
            prompt_candidate += ['\nBrainstormed Answers: ']

            current_question = self.train_promptgenerator._make_prompt(context, question, None, self.train_promptgenerator.BRIEF)
            # few-shot + one new question
            local_prompt = prompt + current_question

            logging.info('P_TRUE >> Current Question: '.ljust(25) + current_question)

            responses = []

            for j in range(num_generations + 1):

                if j == 0:
                    temperature = 0.1
                else:
                    temperature = 1.0

                response, _  = self.model.predict(local_prompt, temperature)
                logging.info('P_TRUE >> Current Response: '.ljust(25) + response)

                responses.append(response)
                prompt_candidate += [f'{response.strip()} \n']
                if j == 0:
                    # Save most likely response and compute correctness metric for it.
                    most_likely_response = response
                    is_correct = self.metric(response, example, self.model)
                    #is_correct = True
                    answers = [answer for answer in example['answers']['text']]
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
            prompt_candidate += [' A' if is_correct else ' B']

            prompt_len = len(self.model.tokenizer.encode(''.join(few_shot_prompt + prompt_candidate)))
            # At test time, get a maximum of `num_generations * model.token_limit` extra tokens
            # 200 buffer for question and 'Possible Answer'.
            max_input_len = prompt_len + num_generations * self.model.max_new_tokens + 200

            if max_input_len < self.model.token_limit:
                few_shot_prompt.extend(prompt_candidate)
            else:
                logging.warning('Cutting of p_true prompt at length %d.', it)
                break

        return ''.join(few_shot_prompt), all_responses, it,p_true_indices





    def calculate_p_true(self,model, question, most_probable_answer, brainstormed_answers,few_shot_prompt, hint=False):
        """Calculate p_true uncertainty metric."""

        if few_shot_prompt:
            prompt = few_shot_prompt + '\n'
        else:
            prompt = ''

        prompt += 'Question: ' + question
        prompt += '\nBrainstormed Answers: '
        for answer in brainstormed_answers + [most_probable_answer]:
            prompt += answer.strip() + '\n'
        prompt += 'Possible answer: ' + most_probable_answer + '\n'
        if not hint:
            prompt += 'Is the possible answer:\n'
            prompt += 'A) True\n'
            prompt += 'B) False\n'
            prompt += 'The possible answer is:'
        else:
            prompt += 'Do the brainstormed answers match the possible answer? Respond with A if they do, if they do not respond with B. Answer:'

        log_prob = model.get_p_true(prompt,"A")

        return log_prob

   

    def get_reference(self,example):
        if 'answers' not in example:
            example = example['reference']
        answers = example['answers']
        answer_starts = answers.get('answer_start', [])
        reference = {'answers': {'answer_start': answer_starts, 'text': answers['text']}, 'id': example['id']}
        return reference


    def save_wandb(self,object, file):
        with open(f'{wandb.run.dir}/{file}', 'wb') as f:
            pickle.dump(object, f)
        wandb.save(f'{wandb.run.dir}/{file}')