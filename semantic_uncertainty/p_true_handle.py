import logging
from uncertainty.uncertainty_measures import p_true as p_true_utils
def p_true_handler(args,model,train_dataset,p_true_indices,prompt,BRIEF,make_prompt,metric):
    p_true_few_shot_prompt, p_true_responses, len_p_true = p_true_utils.construct_few_shot_prompt(
        model=model, dataset=train_dataset, indices=p_true_indices,
        prompt=prompt, brief=BRIEF,
        brief_always=args.brief_always and args.enable_brief,
        make_prompt=make_prompt, num_generations=args.num_generations,
        metric=metric)

    return p_true_few_shot_prompt, p_true_responses, len_p_true
