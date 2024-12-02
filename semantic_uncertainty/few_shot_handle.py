import logging
from uncertainty.utils import utils
def few_shot_handler(args,train_dataset,prompt_indices):
    # Create Few-Shot prompt.
    make_prompt = utils.get_make_prompt(args)
    BRIEF = utils.BRIEF_PROMPTS[args.brief_prompt]
    arg = args.brief_always if args.enable_brief else True
    prompt = utils.construct_fewshot_prompt_from_indices(
        train_dataset, prompt_indices, BRIEF, arg, make_prompt)
    logging.info('Prompt is: %s', prompt)
    return make_prompt,prompt,BRIEF