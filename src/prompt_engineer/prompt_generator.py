import os
import re
import random 
user_tag = "USER:"
assistant_tag = "ASSISTANT:"

class PromptGenerator:
    def __init__(self, config,dataset_item):
        self.prompt_template_path = config['experiment']['prompt_template_path']
        self.prompt_template = self._load_template(self.prompt_template_path)
        self.shots = config['experiment']['few-shot'] 
        self.add_tag = config['experiment']['add_tag']
        self.dataset_item = dataset_item
        self.unused_indices = set(range(len(dataset_item)))


    def generate_prompt_by_id(self,id):
        self._validate_placeholder_existence()
        prompt = ""
        if (self.shots):
            raise ValueError("Unfinshed!!!!")
        else:    
            prompt = self._generate_zeroshot_prompt_by_id(id)
        
        if self.add_tag:
            prompt = self._add_tag_for_prompt(prompt,user_tag,assistant_tag)
        return prompt

    def generate_prompts_by_count(self,count):
        self._validate_placeholder_existence()
        if (self.shots):
            raise ValueError("Unfinshed!!!!")
        else:    
            prompts = self._generate_zeroshot_prompts_by_count(count)
        return prompts



    def get_unused_indices(self):
        return list(self.unused_indices)


    def _load_template(self, template_path):

        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file not found at: {template_path}")
        
        with open(template_path, 'r', encoding='utf-8') as file:
            template = file.read()
        
        return template



    def _generate_zeroshot_prompt_by_id(self, id):

        if id not in self.unused_indices:
            raise ValueError(f"ID {id} has already been used for prompt generation, or this is an illegal ID")

        placeholders = self._get_replaced_words()
        item = self.dataset_item[id]
        prompt = self.prompt_template

        for placeholder in placeholders:
            pattern = r'\$\{' + re.escape(placeholder) + r'\}'
            replacement = str(item.get(placeholder.lower(), ''))
            prompt = re.sub(pattern, replacement, prompt)

        if placeholders:
            self.unused_indices.discard(id) 
        return prompt



    def _generate_zeroshot_prompts_by_count(self, count):
        if count > len(self.unused_indices):
            raise ValueError("The requested number exceeds the available unused IDs.")

        selected_ids = random.sample(list(self.unused_indices), count)
        
        prompts = []
        
        for id in selected_ids:
            try:
                prompt = self.generate_prompt_by_id(id)
                prompts.append(prompt)
            except ValueError as e:
                print(f"Error when making prompt through id {id}: {e}")

        self.unused_indices.difference_update(selected_ids)
        return prompts

    def _get_replaced_words(self):
        placeholders = re.findall(r'\$\{(\w+)\}', self.prompt_template)
        return placeholders
    

    def _validate_placeholder_existence(self):
        placeholders = self._get_replaced_words()  
        placeholders = [placeholder.lower() for placeholder in re.findall(r'\$\{(\w+)\}', self.prompt_template)]

        missing_placeholders = []
        for item in self.dataset_item:
            for placeholder in placeholders:
                if placeholder not in item:
                    missing_placeholders.append(placeholder)

        if missing_placeholders:
            missing_placeholders_str = ', '.join(set(missing_placeholders)) 
            raise ValueError(f"Missing required placeholders in dataset: {missing_placeholders_str}")




    def _add_tag_for_prompt(self,prompt,user_tag,assistant_tag):
        tagged_prompt = f"{user_tag} {prompt}\n{assistant_tag}"
        return tagged_prompt
        
   
    



