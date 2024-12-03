import logging
def few_shot_handler(args,make_prompt,BRIEF,train_dataset,prompt_indices):
    # Create Few-Shot prompt.
    arg = args.brief_always if args.enable_brief else True
    prompt = utils.construct_fewshot_prompt_from_indices(
        train_dataset, prompt_indices, BRIEF, arg, make_prompt)
    logging.info('Prompt is: %s', prompt)
    return prompt



def few_shot(args, train_dataset, prompt_indices):
    prompt = {}
    hint = "Please select the most correct option"
    prompt["hint"] = hint
    prompt["images"] = []  
    prompt["questions"] = []  
    prompt["answers"] = []  
    prompt['options'] = []  

    for idx in prompt_indices:
        image = Image.open(train_dataset[idx]["original_image"]).convert("RGB")
        prompt["images"].append(image)
        prompt["questions"].append(train_dataset[idx]["question"])
        prompt["answers"].append(train_dataset[idx]["answers"])
        prompt['options'].append(train_dataset[idx]["Options"])
    
    return prompt
