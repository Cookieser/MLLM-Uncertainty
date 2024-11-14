import os
import re
class PromptGenerator:
    def __init__(self, config,dataset_item):
        self.prompt_template_path = config['experiment']['prompt_template_path']
        self.prompt_template = self._load_template(self.prompt_template_path)
        self.shots = config['experiment']['few-shot'] 
        self.dataset_item = dataset_item


    def generate_prompt_by_id(self,id):
        self._validate_placeholder_existence()
        if (self.shots):
            raise ValueError("Unfinshed!!!!")
        else:    
            return self._generate_zeroshot_prompt_by_id(id)



    def _load_template(self, template_path):

        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file not found at: {template_path}")
        
        with open(template_path, 'r', encoding='utf-8') as file:
            template = file.read()
        
        return template


    def _generate_zeroshot_prompt_by_id(self, id):
        placeholders = self._get_replaced_words()
        item = self.dataset_item[id]
        prompt = self.prompt_template

        for placeholder in placeholders:
            pattern = r'\$\{' + re.escape(placeholder) + r'\}'
            replacement = str(item.get(placeholder.lower(), ''))
            prompt = re.sub(pattern, replacement, prompt)

        return prompt



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
    
    #def remain_index
    #def remain_number
    #一部分被用来生产prompt了，一部分用来生成shot了
    #写一个能直接批量生产prompts们的函数
    #是否重复提示词也是一个变量