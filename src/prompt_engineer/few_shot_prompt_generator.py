# from prompt_generator import PromptGenerator

# class FewShotPromptGenerator(PromptGenerator):
#     def __init__(self, config, dataset_item):
#         super().__init__(config, dataset_item)
#         #self.brief_always = config['experiment']['brief_always']

#     def make_example(self):
#             example = ''
#             if brief_always:
#                 prompt += brief
#             if args.use_context and (context is not None):
#                 prompt += f"Context: {context}\n"
#             prompt += f"Question: {question}\n"
#             if answer:
#                 prompt += f"Answer: {answer}\n\n"
#             else:
#                 prompt += 'Answer:'
#             return prompt
#     else:
#         raise ValueError

#     return make_prompt


#     def generate_fewshot_prompts(self, nums_shot,nums_prompts):


# 注意：unused_indices的变化，一部分被用来生产prompt，一部分用来生成shot
# 注意变量：是否重复提示词/few-shot number/prompt num  都是变量