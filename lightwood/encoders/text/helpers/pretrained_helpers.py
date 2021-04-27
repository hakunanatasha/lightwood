"""
2021.04.26

Adding a binning function to test quantile regression.
Currently in quantile form, might be also worth treating in log-form

2021.03.05

Basic helper functions for PretrainedLang
"""
import torch
from transformers import AdamW
import numpy as np

def bin_targets(targets, quantiles):
    """
    Given a regression input, convert into digitized bins
    based on quantiles of the input distribution:

    Args:
    :param targets, list[int]; list of input distribution of target variable
    :param quantiles np.ndarray; set of percentiles from [0, 1]

    """
    pscores = [np.quantile(targets, q) for q in quantiles]

    # Add a little extra to the ends to bin min/max values appropriately
    pscores[0] = pscores[0] - np.abs(pscores[0]*0.05)
    pscores[-1] = pscores[-1] + np.abs(pscores[-1]*0.05)

    # For each element in the targets, digitize and return
    # the bin it would belong in
    return np.digitize(targets, pscores)

class TextEmbed(torch.utils.data.Dataset):
    """
    Dataset class for quick embedding/label retrieval.
    Labels should be in the index space.

    If the labels provided are not in torch form, will convert them.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def train_model(model, dataset, device, scheduler=None, log=None, optim=None, n_epochs=4):
    """
    Generic training function, given an arbitrary model.
    
    Given a model, train for n_epochs.

    model - torch.nn model;
    dataset - torch.DataLoader; dataset to train
    device - torch.device; cuda/cpu
    log - lightwood.logger.log; print output
    optim - transformers.optimization.AdamW; optimizer
    n_epochs - number of epochs to train

    """
    losses = []
    model.train()
    if optim is None:
        optim = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(n_epochs):
        total_loss = 0
        for batch in dataset:
            optim.zero_grad() 

            inpids = batch['input_ids'].to(device)
            attn = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(inpids, attention_mask=attn, labels=labels)
            loss = outputs[0]

            total_loss += loss.item()
            
            loss.backward()
            optim.step()

            if scheduler is not None:
                scheduler.step()

        print("Epoch", epoch+1, "Loss", total_loss)
    return model, losses