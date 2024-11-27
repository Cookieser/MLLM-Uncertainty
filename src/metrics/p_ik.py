"""Predict model correctness from linear classifier."""
import logging
import torch
import wandb

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def get_p_ik(train_embeddings, is_false, eval_embeddings=None, eval_is_false=None):
    """Fit linear classifier to embeddings to predict model correctness."""

    logging.info('Accuracy of model on Task: %f.', 1 - torch.tensor(is_false).mean())  # pylint: disable=no-member

    # Convert the list of tensors to a 2D tensor.
    train_embeddings_tensor = torch.cat(train_embeddings, dim=0)  # pylint: disable=no-member
    # Convert the tensor to a numpy array.
    embeddings_array = train_embeddings_tensor.cpu().numpy()

    # Split the data into training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(  # pylint: disable=invalid-name
        embeddings_array, is_false, test_size=0.2, random_state=42)  # pylint: disable=invalid-name

    # Fit a logistic regression model.
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict deterministically and probabilistically and compute accuracy and auroc for all splits.
    X_eval = torch.cat(eval_embeddings, dim=0).cpu().numpy()  # pylint: disable=no-member,invalid-name
    y_eval = eval_is_false

    Xs = [X_train, X_test, X_eval]  # pylint: disable=invalid-name
    ys = [y_train, y_test, y_eval]  # pylint: disable=invalid-name
    suffixes = ['train_train', 'train_test', 'eval']

    metrics, y_preds_proba = {}, {}

    for suffix, X, y_true in zip(suffixes, Xs, ys):  # pylint: disable=invalid-name

        # If suffix is eval, we fit a new model on the entire training data set
        # rather than just a split of the training data set.
        if suffix == 'eval':
            model = LogisticRegression()
            model.fit(embeddings_array, is_false)
            convergence = {
                'n_iter': model.n_iter_[0],
                'converged': (model.n_iter_ < model.max_iter)[0]}

        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)
        y_preds_proba[suffix] = y_pred_proba
        acc_p_ik_train = accuracy_score(y_true, y_pred)
        auroc_p_ik_train = roc_auc_score(y_true, y_pred_proba[:, 1])
        split_metrics = {
            f'acc_p_ik_{suffix}': acc_p_ik_train,
            f'auroc_p_ik_{suffix}': auroc_p_ik_train}
        metrics.update(split_metrics)

    logging.info('Metrics for p_ik classifier: %s.', metrics)
    wandb.log({**metrics, **convergence})

    # Return model predictions on the eval set.
    return y_preds_proba['eval'][:, 1]




# if args.compute_p_ik or args.compute_p_ik_answerable:
#     # Assemble training data for embedding classification.
#     train_is_true, train_embeddings, train_answerable = [], [], []
#     for tid in train_generations:
#         most_likely_answer = train_generations[tid]['most_likely_answer']
#         train_embeddings.append(most_likely_answer['embedding'])
#         train_is_true.append(most_likely_answer['accuracy'])
#         train_answerable.append(is_answerable(train_generations[tid]))
#     train_is_false = [0.0 if is_t else 1.0 for is_t in train_is_true]
#     train_unanswerable = [0.0 if is_t else 1.0 for is_t in train_answerable]
#     logging.info('Unanswerable prop on p_ik training: %f', np.mean(train_unanswerable))

# if args.compute_p_ik:
#     logging.info('Starting training p_ik on train embeddings.')
#     # Train classifier of correct/incorrect from embeddings.
#     p_ik_predictions = get_p_ik(
#         train_embeddings=train_embeddings, is_false=train_is_false,
#         eval_embeddings=validation_embeddings, eval_is_false=validation_is_false)
#     result_dict['uncertainty_measures']['p_ik'] = p_ik_predictions
#     logging.info('Finished training p_ik on train embeddings.')

# if args.compute_p_ik_answerable:
#     # Train classifier of answerable/unanswerable.
#     p_ik_predictions = get_p_ik(
#         train_embeddings=train_embeddings, is_false=train_unanswerable,
#         eval_embeddings=validation_embeddings, eval_is_false=validation_unanswerable)
#     result_dict['uncertainty_measures']['p_ik_unanswerable'] = p_ik_predictions