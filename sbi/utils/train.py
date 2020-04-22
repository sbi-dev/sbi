import torch
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from torch import optim
from sbi.utils.torchutils import get_default_device
from torch.nn.utils import clip_grad_norm_
from copy import deepcopy

def train_uncondtional(
        flow,
        dataset,
        batch_size=100,
        learning_rate=5e-4,
        validation_fraction=0.1,
        stop_after_epochs=20,
        clip_grad_norm=True
        ,):
    """
    Train a uncondtional normalizing flow with maximum likelihood p(x)
    Args:
        flow: nflows.flows.Flow
        dataset: torch.tensor(). Data x.
        batch_size: size of the minibatch
        learning_rate: learning rate
        validation_fraction: fraction of datapoints to be used for validation
        stop_after_epochs: stop training after validation loss has not decreased for this many epochs
        clip_grad_norm: whether to clip the norm of the gradient
    Returns: trained flow
    """

    # Get total number of training examples.
    num_examples = dataset.shape[0]

    # Select random neural_net and validation splits from (parameter, observation) pairs.
    permuted_indices = torch.randperm(num_examples)
    num_training_examples = int((1 - validation_fraction) * num_examples)
    num_validation_examples = num_examples - num_training_examples
    train_indices, val_indices = (
        permuted_indices[:num_training_examples],
        permuted_indices[num_training_examples:],
    )

    device = get_default_device()

    dataset = dataset.to(device)
    # Dataset is shared for training and validation loaders.
    dataset = data.TensorDataset(dataset)
    flow = flow.to(device)

    # Create neural_net and validation loaders using a subset sampler.
    train_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        sampler=SubsetRandomSampler(train_indices),
    )
    val_loader = data.DataLoader(
        dataset,
        batch_size=min(batch_size, num_validation_examples),
        shuffle=False,
        drop_last=True,
        sampler=SubsetRandomSampler(val_indices),
    )

    optimizer = optim.Adam(
        list(flow.parameters()), lr=learning_rate,
    )
    # Keep track of best_validation log_prob seen so far.
    best_validation_log_prob = -1e100
    # Keep track of number of epochs since last improvement.
    epochs_since_last_improvement = 0
    # Keep track of model with best validation performance.
    best_model_state_dict = None

    # Each run also has a dictionary of summary statistics which are populated
    # over the course of training.
    summary = {
        "epochs": [],
        "best-validation-log-probs": [],
    }

    epochs = 0
    converged = False
    while not converged:

        # Train for a single epoch.
        flow.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = (
                batch[0],#.to(device),
            )

            # just do maximum likelihood
            log_prob = flow.log_prob(
                inputs[0]
            )
            loss = -torch.mean(log_prob)
            loss.backward()
            if clip_grad_norm:
                clip_grad_norm_(flow.parameters(), max_norm=5.0)
            optimizer.step()

        epochs += 1

        # Calculate validation performance.
        flow.eval()
        log_prob_sum = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = (
                    batch[0].to(device),
                )
                # just do maximum likelihood in the first round
                log_prob = flow.log_prob(
                    inputs[0]
                )
                log_prob_sum += log_prob.sum().item()
        validation_log_prob = log_prob_sum / num_validation_examples

        print('Epoch:', epochs, ' --  validation loss', -validation_log_prob)

        # Check for improvement in validation performance over previous epochs.
        if validation_log_prob > best_validation_log_prob:
            best_validation_log_prob = validation_log_prob
            epochs_since_last_improvement = 0
            best_model_state_dict = deepcopy(flow.state_dict())
        else:
            epochs_since_last_improvement += 1

        # If no validation improvement over many epochs, stop training.
        if epochs_since_last_improvement > stop_after_epochs - 1:
            flow.load_state_dict(best_model_state_dict)
            converged = True

    # Update summary.
    summary["epochs"].append(epochs)
    summary["best-validation-log-probs"].append(best_validation_log_prob)

    return flow