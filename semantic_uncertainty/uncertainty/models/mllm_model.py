import requests
import logging
from PIL import Image
from uncertainty.models.base_model import BaseModel
from uncertainty.models.base_model import STOP_SEQUENCES
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from urllib.parse import urlparse
from pprint import pprint
import torch.nn.functional as F
import os 

class MLLMModel(BaseModel):
    def __init__(self, model_name, stop_sequences=None, max_new_tokens=None):
        if max_new_tokens is None:
            raise
        self.max_new_tokens = max_new_tokens

        if stop_sequences == 'default':
            stop_sequences = STOP_SEQUENCES

        if (model_name =='llava-1.5-7b-hf'):
            model_id = "llava-hf/llava-1.5-7b-hf"
            self.model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True,device_map='auto')
            self.processor = AutoProcessor.from_pretrained(model_id)
        else:
            raise ValueError
        
        self.model_name = model_name
        

    def predict(self,question,image,temperature, return_full=False):
        conversation = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image"},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = self.processor(images=image, text=prompt, return_tensors='pt')
        with torch.no_grad():
            output = self.model.generate(
                **inputs, 
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=temperature,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
                )
        full_answer = self.processor.decode(output.sequences[0], skip_special_tokens=True)
        generated_tokens = output.sequences[0].tolist()

        eos_token_id = self.processor.tokenizer.eos_token_id
        eos_token = self.processor.tokenizer.decode([eos_token_id])



        if "ASSISTANT:" in full_answer:
            clean_answer = full_answer.split("ASSISTANT:")[-1].strip()
        else:
            clean_answer = full_answer.strip() 

        if clean_answer.endswith(eos_token):
            clean_answer = clean_answer[: -len(eos_token)].strip()
        
        clean_answer_tokens = self.processor.tokenizer.encode(clean_answer, add_special_tokens=False)

        hidden_states = output.hidden_states  # List of tensors for each layer

        
        if len(hidden_states) == 0:
            logging.warning('Nothing happens! ')
        elif (len(hidden_states) == 1):
            logging.warning('Taking first and only generation for hidden! ')
            last_input = hidden_states[0]
        else:
            last_input = hidden_states[-2]
        
        last_layer = last_input[-1]
        last_token_embedding = last_layer[:, -1, :].cpu()
        
        
        # Compute transition scores log-likelihoods
        transition_scores = self.model.compute_transition_scores(
            output.sequences, output.scores, normalize_logits=True
        )
        log_likelihoods = [score.item() for score in transition_scores[0]]

        # Extract log-likelihoods matching clean_answer_tokens
        clean_log_likelihoods = log_likelihoods[-len(clean_answer_tokens):]



        if return_full:
            return full_answer

        return clean_answer,clean_log_likelihoods,last_token_embedding

    def get_p_true(self,conversation,images):

        target_token = "A"
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = self.processor(images=images, text=prompt, return_tensors='pt')
        with torch.no_grad():
            output = self.model.generate(
                **inputs, 
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=False,
                )
        full_answer = self.processor.decode(output.sequences[0], skip_special_tokens=True)
        # print(full_answer)
        generated_tokens = output.sequences[0].tolist()

        eos_token_id = self.processor.tokenizer.eos_token_id
        eos_token = self.processor.tokenizer.decode([eos_token_id])



        if "ASSISTANT:" in full_answer:
            clean_answer = full_answer.split("ASSISTANT:")[-1].strip()
        else:
            clean_answer = full_answer.strip() 

        if clean_answer.endswith(eos_token):
            clean_answer = clean_answer[: -len(eos_token)].strip()
        
        clean_answer_tokens = self.processor.tokenizer.encode(clean_answer, add_special_tokens=False)
        
        logits_for_token = output.scores[0][-len(clean_answer_tokens)] 

        top_token_id = torch.argmax(logits_for_token).item()  # Find the token ID with the highest probability

        # Decode the token ID to text
        decoded_token = self.processor.tokenizer.decode([top_token_id])

        logging.info(f"==> Decoded token: {decoded_token}")
        logging.info(f"==> Target token: {target_token}")
 
        target_token_id = self.processor.tokenizer.encode(target_token, add_special_tokens=False)[0]

        probs = F.softmax(logits_for_token, dim=-1)  
        log_prob_a = torch.log(probs[target_token_id])  

        return log_prob_a.item()


    
    def predict_few_shot(self,question,image,options,few_shot_info,temperature, return_full=False):
        hint = few_shot_info.get("hint", "")
        few_shot_images = few_shot_info.get("images", [])
        few_shot_questions = few_shot_info.get("questions", [])
        few_shot_answers = few_shot_info.get("answers", [])
        few_shot_options = few_shot_info.get("options", [])

        few_shot_options = [" "] * len(few_shot_questions) if not few_shot_options else few_shot_options

        conversation = []
       

        for fs_question, fs_option, fs_answers in zip(few_shot_questions,few_shot_options, few_shot_answers):
            #print(f"Original fs_answers: {fs_answers}, Type: {type(fs_answers)}")
            if isinstance(fs_answers, list):
                fs_answers = ", ".join(fs_answers)  
            elif not isinstance(fs_answers, str):
                raise ValueError(f"fs_answers must be a string or list of strings, but got {type(fs_answers)}")
            conversation.append( {
                "role": "user",
                "content": [
                    {"type": "text", "text": hint},
                    {"type": "text", "text": fs_question},
                    {"type": "text", "text": fs_option},   
                    {"type": "image"},  
                    
                ],
            })

            conversation.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": fs_answers},
     
                    ],
            })
    
        options = " " if not options else options

        # Add the current question and image
        conversation.append( {
            "role": "user",
            "content": [
                {"type": "text", "text": hint},
                {"type": "text", "text": question},
                {"type": "text", "text": options},
                {"type": "image"}, 
                {"type": "text", "text": "\n"},
            ],
        })
        images = few_shot_images + [image]

        #pprint(conversation)    

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)


        few_shot_images_count = len(few_shot_images)


        prompt_images_num = len(few_shot_questions) + 1


        assert few_shot_images_count + 1 == prompt_images_num, (
    f"Inconsistent counts: Number of images ({few_shot_images_count + 1}) "
    f"does not match number of prompt images ({prompt_images_num})."
)

        inputs = self.processor(images=images, text=prompt, return_tensors='pt')
        with torch.no_grad():
            output = self.model.generate(
                **inputs, 
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=temperature,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
                )
        full_answer = self.processor.decode(output.sequences[0], skip_special_tokens=True)
        generated_tokens = output.sequences[0].tolist()

        eos_token_id = self.processor.tokenizer.eos_token_id
        eos_token = self.processor.tokenizer.decode([eos_token_id])



        if "ASSISTANT:" in full_answer:
            clean_answer = full_answer.split("ASSISTANT:")[-1].strip()
        else:
            clean_answer = full_answer.strip() 

        if clean_answer.endswith(eos_token):
            clean_answer = clean_answer[: -len(eos_token)].strip()
        
        clean_answer_tokens = self.processor.tokenizer.encode(clean_answer, add_special_tokens=False)

        hidden_states = output.hidden_states  # List of tensors for each layer

        
        if len(hidden_states) == 0:
            logging.warning('Nothing happens! ')
        elif (len(hidden_states) == 1):
            logging.warning('Taking first and only generation for hidden! ')
            last_input = hidden_states[0]
        else:
            last_input = hidden_states[-2]
        
        last_layer = last_input[-1]
        last_token_embedding = last_layer[:, -1, :].cpu()
        
        
        # Compute transition scores log-likelihoods
        transition_scores = self.model.compute_transition_scores(
            output.sequences, output.scores, normalize_logits=True
        )
        log_likelihoods = [score.item() for score in transition_scores[0]]

        # Extract log-likelihoods matching clean_answer_tokens
        clean_log_likelihoods = log_likelihoods[-len(clean_answer_tokens):]



        if return_full:
            return full_answer

        return clean_answer,clean_log_likelihoods,last_token_embedding

   
    
