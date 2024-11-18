import os
import re
import random 
user_tag = "USER:"
assistant_tag = "ASSISTANT:"


BRIEF_PROMPTS = {
    'default': "Answer the following question as briefly as possible.\n",
    'chat': 'Answer the following question in a single brief but complete sentence.\n'}

class PromptGenerator:
    def __init__(self, config,dataset_item):
        self.prompt_template_path = config['prompt']['prompt_template_path']
        self.prompt_template = self._load_template(self.prompt_template_path)
        self.shots = config['prompt']['few-shot'] 
        self.shot_num = config['prompt']['shot_num'] 
        self.add_tag = config['prompt']['add_tag']
        self.dataset_item = dataset_item
        self.unused_indices = set(range(len(dataset_item)))
        self.brief_always = config['prompt']['brief_always'] 
        self.use_contexts = config['prompt']['use_context']


    def generate_template_prompt_by_id(self,id):
        self._validate_placeholder_existence()

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
        
        if self.add_tag:
            prompt = self._add_tag_for_prompt(prompt,user_tag,assistant_tag)
        return prompt

    def generate_template_prompts_by_count(self,count):
        self._validate_placeholder_existence()
         
        if count > len(self.unused_indices):
            raise ValueError("The requested number exceeds the available unused IDs.")

        selected_ids = random.sample(list(self.unused_indices), count)
        
        prompts = []
        
        for id in selected_ids:
            try:
                prompt = self.generate_template_prompt_by_id(id)
                prompts.append(prompt)
            except ValueError as e:
                print(f"Error when making prompt through id {id}: {e}")

        self.unused_indices.difference_update(selected_ids)

        return prompts
    
    def generate_template_prompts_from_indices(self, indices):
        self._validate_placeholder_existence()
        
        # Validate that all requested indices are in the unused set
        if not set(indices).issubset(self.unused_indices):
            raise ValueError("Some of the provided indices are not available in the unused set.")
        
        prompts = []
        
        for index in indices:
            try:
                prompt = self.generate_template_prompt_by_id(index)
                prompts.append(prompt)
            except ValueError as e:
                print(f"Error when making prompt through index {index}: {e}")
        
        # Remove the used indices from the unused set
        self.unused_indices.difference_update(indices)
        
        return prompts



    def get_unused_indices(self):
        return list(self.unused_indices)


    def _load_template(self, template_path):

        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file not found at: {template_path}")
        
        with open(template_path, 'r', encoding='utf-8') as file:
            template = file.read()
        
        return template


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
    


    def _make_prompt(self,context, question, answer, brief,brief_always):
        prompt = ''
        if brief_always:
            prompt += brief
        if self.use_contexts and (context is not None):
            prompt += f"Context: {context}\n"
        prompt += f"Question: {question}\n"
        if answer:
            prompt += f"Answer: {answer}\n\n"
        else:
            prompt += 'Answer:'
        return prompt



    def construct_fewshot_prompt_from_indices(self, brief= BRIEF_PROMPTS['default']):
        """Given a dataset and indices, construct a fewshot prompt."""
        example_indices = random.sample(list(self.unused_indices), self.shot_num)
        if not self.brief_always:
            prompt = brief
        else:
            prompt = ''

        # Validate that all requested indices are in the unused set
        if not set(example_indices).issubset(self.unused_indices):
            raise ValueError("Some of the provided indices are not available in the unused set.")

        for example_index in example_indices:

            example = self.dataset_item[example_index]
            context = example["context"]
            question = example["question"]
            answer = example["answers"]["text"][0]

            prompt = prompt + self._make_prompt(context, question, answer, brief,self.brief_always)

        self.unused_indices.difference_update(example_indices)

        return prompt
   