import copy
import logging
from collections import Counter
import torch
import os

import accelerate

from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers import StoppingCriteria
from transformers import StoppingCriteriaList
from huggingface_hub import snapshot_download

from .base_model import BaseModel
from .base_model import STOP_SEQUENCES

class StoppingCriteriaSub(StoppingCriteria):
    """Stop generations when they match a particular text or token."""
    def __init__(self, stops, tokenizer, match_on='text', initial_length=None):
        super().__init__()
        self.stops = stops
        self.initial_length = initial_length
        self.tokenizer = tokenizer
        self.match_on = match_on
        if self.match_on == 'tokens':
            self.stops = [torch.tensor(self.tokenizer.encode(i)).to('cuda') for i in self.stops]
            print(self.stops)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        del scores  # `scores` arg is required by StoppingCriteria but unused by us.
        #generation = self.tokenizer.decode(input_ids[0][self.initial_length:], skip_special_tokens=False)
        #print("Current generation:", generation)  # Debugging print
        for stop in self.stops:
            if self.match_on == 'text':
                generation = self.tokenizer.decode(input_ids[0][self.initial_length:], skip_special_tokens=False)
                match = stop in generation
                #print("Checking text stop:", stop, "Match:", match) 
            elif self.match_on == 'tokens':
                # Can be dangerous due to tokenizer ambiguities.
                match = stop in input_ids[0][-len(stop):]
                #print("Checking token stop:", stop, "Match:", match)
            else:
                raise ValueError("Unknown match_on type:", self.match_on)
            if match:
                #print("Stopping criteria met with:", stop)
                return True
        return False




class HuggingfaceModel(BaseModel):

    def __init__(self, config):
        self.model_name = config['model']['model_name']
        self.max_new_tokens = config['model']['max_new_tokens']
        self.stop_sequences = config['model']['stop_sequences']
        

        if self.stop_sequences == 'default':
            self.stop_sequences = STOP_SEQUENCES

        base_model_path = f"{self.model_name}"
        self.model = AutoModelForCausalLM.from_pretrained(base_model_path,torch_dtype=torch.float16,device_map="auto")
        use_fast_tokenizer = "LlamaForCausalLM" not in self.model.config.architectures
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path,use_fast=use_fast_tokenizer, padding_side="left", legacy=False, token_type_ids=None)

        self.token_limit = 4096 if 'Llama-2' in self.model_name else 2048
        self.stop_sequences = self.stop_sequences + [self.tokenizer.eos_token]
        
        


    def predict(self, input_data, temperature=1.0,device='cuda',return_full=False):

        inputs = self.tokenizer(input_data, return_tensors="pt").to(device)

        if 'llama' in self.model_name.lower() or 'falcon' in self.model_name or 'mistral' in self.model_name.lower():
            if 'token_type_ids' in inputs:  # Some HF models have changed.
                del inputs['token_type_ids']
            pad_token_id = self.tokenizer.eos_token_id
        else:
            pad_token_id = None

        if self.stop_sequences is not None:
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
                stops=self.stop_sequences,
                initial_length=len(inputs['input_ids'][0]),
                tokenizer=self.tokenizer)])
        else:
            stopping_criteria = None

        logging.debug('temperature: %f', temperature)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                return_legacy_cache=False,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
                temperature=temperature,
                do_sample=True,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                
            )

        if len(outputs.sequences[0]) > self.token_limit:
            raise ValueError(
                'Generation exceeding token limit %d > %d',
                len(outputs.sequences[0]), self.token_limit)

        full_answer = self.tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True)

        if return_full:
            return full_answer

        #print("-----------------------------------")

        #print(full_answer)

        #print("-----------------------------------")

        if full_answer.startswith(input_data):
            input_data_offset = len(input_data)
        else:
            raise ValueError('Have not tested this in a while.')

        # Remove input from answer.
        answer = full_answer[input_data_offset:]

        # Remove stop_words from answer.
        stop_at = len(answer)
        sliced_answer = answer
        if self.stop_sequences is not None:
            for stop in self.stop_sequences:
                if answer.endswith(stop):
                    stop_at = len(answer) - len(stop)
                    sliced_answer = answer[:stop_at]
                    break
            if not all([stop not in sliced_answer for stop in self.stop_sequences]):
                error_msg = 'Error: Stop words not removed successfully!'
                error_msg += f'Answer: >{answer}< '
                error_msg += f'Sliced Answer: >{sliced_answer}<'
                if 'falcon' not in self.model_name.lower():
                    raise ValueError(error_msg)
                else:
                    logging.error(error_msg)

        # Remove whitespaces from answer (in particular from beginning.)
        sliced_answer = sliced_answer.strip()

        token_stop_index = self.tokenizer(full_answer[:input_data_offset + stop_at], return_tensors="pt")['input_ids'].shape[1]
        n_input_token = len(inputs['input_ids'][0])
        n_generated = token_stop_index - n_input_token

        if n_generated == 0:
            logging.warning('Only stop_words were generated. For likelihoods and embeddings, taking stop word instead.')
            n_generated = 1
       


        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True)
        # Transition_scores[0] only contains the scores for the first generated tokens.

        log_likelihoods = [score.item() for score in transition_scores[0]]
        if len(log_likelihoods) == 1:
            print('Taking first and only generation for log likelihood!')
            log_likelihoods = log_likelihoods
        else:
            log_likelihoods = log_likelihoods[:n_generated]

        if len(log_likelihoods) == self.max_new_tokens:
            print('Generation interrupted by max_token limit.')

        if len(log_likelihoods) == 0:
            raise ValueError

        return sliced_answer,log_likelihoods

    def get_p_true(self, input_data, answer="A"):

        input_data += f" {answer}"
        print(input_data)
    
        tokenized_prompt_true = self.tokenizer(
            input_data, 
            return_tensors='pt',
            ).to('cuda')['input_ids']
    
        target_ids_true = tokenized_prompt_true.clone()
    
        answer_length = len(self.tokenizer(answer, return_tensors='pt')['input_ids'][0])

        target_ids_true[0, :-answer_length] = -100

        with torch.no_grad():
            model_output_true = self.model(tokenized_prompt_true, labels=target_ids_true)

        loss_true = model_output_true.loss
        return -loss_true.item()

