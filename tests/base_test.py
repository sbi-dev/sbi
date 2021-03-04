import pytest
import torch
from torch.utils.data import TensorDataset

from sbi.inference import SNPE


@pytest.mark.parametrize("training_batch_size", (1, 10, 100))
def test_get_dataloaders(training_batch_size):

    N = 1000
    validation_fraction = 0.1

    dataset = TensorDataset(torch.ones(N), torch.zeros(N))

    inferer = SNPE()

    _, val_loader = inferer.get_dataloaders(
        dataset,
        training_batch_size=training_batch_size,
        validation_fraction=validation_fraction,
    )

    assert len(val_loader) * val_loader.batch_size == int(validation_fraction * N)
