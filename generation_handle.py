import logging
from tqdm import tqdm
from uncertainty.utils import utils
from uncertainty.uncertainty_measures import p_true as p_true_utils
# Sample Process: Use to sample many times in different t
def generation_handler(args,dataset,dataset_split,indices,make_prompt,BRIEF,prompt,p_true_few_shot_prompt,model,metric,accuracies, generations,  p_trues):
    it = 0
    for index in tqdm(indices):
        if (it + 1 % 10) == 0:
            gc.collect()
            torch.cuda.empty_cache()
        it += 1

        # Grab example at index.
        example = dataset[index]
        question, context = example["question"], example['context']
        generations[example['id']] = {'question': question, 'context': context}
        correct_answer = example['answers']['text']

        current_input = make_prompt(
            context, question, None, BRIEF, args.brief_always and args.enable_brief)
        local_prompt = prompt + current_input

        logging.info('Current input: '.ljust(15) + current_input)

        full_responses = []

        # We sample one low temperature answer on which we will compute the
        # accuracy and args.num_generation high temperature answers which will
        # be used to estimate the entropy variants.

        if dataset_split == 'train' and args.get_training_set_generations_most_likely_only:
            num_generations = 1
        else:
            num_generations = args.num_generations + 1

        for i in range(num_generations):

            # Temperature for first generation is always `0.1`.
            temperature = 0.1 if i == 0 else args.temperature

            predicted_answer, token_log_likelihoods, embedding = model.predict(
                local_prompt, temperature)
            embedding = embedding.cpu() if embedding is not None else None

            # Only compute accuracy if question is answerable.
            compute_acc = args.compute_accuracy_at_all_temps or (i == 0)
            if correct_answer and compute_acc:
                acc = metric(predicted_answer, example, model)
            else:
                acc = 0.0  # pylint: disable=invalid-name

            if i == 0:
                logging.info('Iteration ' + str(it) + ':  ' + 80*'#')
                if args.use_context:
                    logging.info('context: '.ljust(15) + str(context))
                logging.info('question: '.ljust(15) + question)
                logging.info('low-t prediction: '.ljust(15) + predicted_answer)
                logging.info('correct answer: '.ljust(15) + str(correct_answer))
                logging.info('accuracy: '.ljust(15) + str(acc))

                accuracies.append(acc)
                most_likely_answer_dict = {
                    'response': predicted_answer,
                    'token_log_likelihoods': token_log_likelihoods,
                    'embedding': embedding,
                    'accuracy': acc}
                generations[example['id']].update({
                    'most_likely_answer': most_likely_answer_dict,
                    'reference': utils.get_reference(example)})

            else:
                logging.info('high-t prediction '.ljust(15) + str(i) + ' : ' + predicted_answer)
                # Aggregate predictions over num_generations.
                full_responses.append(
                    (predicted_answer, token_log_likelihoods, embedding, acc))

        # Append all predictions for this example to `generations`.
        generations[example['id']]['responses'] = full_responses

        if args.compute_p_true and dataset_split == 'validation':
            # Already compute p_true here. Avoid cost of generations in compute_uncertainty script.
            p_true = p_true_utils.calculate_p_true(
                model, question, most_likely_answer_dict['response'],
                [r[0] for r in full_responses], p_true_few_shot_prompt,
                hint=args.p_true_hint)
            p_trues.append(p_true)
            logging.info('p_true: %s', p_true)
    return generations,accuracies,p_trues







            


            


        




