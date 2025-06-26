'''
This code is cloned from RSCFed.(https://github.com/xmed-lab/RSCFed)
Special thanks to the original authors for making their work publicly available.
'''

import numpy as np
import torch
import torch.optim
import copy
from networks.models import ModelFedCon
import torch.nn
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import copy
import torch
import torch.optim
import torch.nn as nn
import numpy as np
from networks.models import ModelFedCon

class SupervisedLocalUpdate(object):
    def __init__(self, args, idxs, n_classes):
        """
        Initializes the supervised client.

        Args:
            args: Command-line arguments or a config object.
            idxs (list): Data indices for this client.
            n_classes (int): Number of classes in the dataset.
        """
        self.args = args
        self.data_idx = idxs
        self.max_grad_norm = args.max_grad_norm

        # Initialize the model once
        net = ModelFedCon(args.model, args.out_dim, n_classes=n_classes, drop_rate=args.drop_rate)
        if len(args.gpu.split(',')) > 1:
            net = torch.nn.DataParallel(net)
        self.model = net.cuda()

        # Initialize the optimizer once
        if args.opt == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.base_lr,
                                              betas=(0.9, 0.999), weight_decay=5e-4)
        elif args.opt == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.base_lr,
                                             momentum=0.9, weight_decay=5e-4)
        elif args.opt == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.base_lr,
                                               weight_decay=0.02)
        
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self, net_w, op_dict, dataloader):
        """
        Performs one round of local supervised training.

        Args:
            net_w (dict): State dictionary of the global model from the server.
            op_dict (dict): State dictionary of the optimizer.
            dataloader (DataLoader): The client's local data loader.

        Returns:
            tuple: A tuple containing:
                - dict: The updated local model's state dictionary.
                - float: The average loss for the training epochs.
                - dict: The updated optimizer's state dictionary.
        """
        # Load the global model state and optimizer state
        self.model.load_state_dict(copy.deepcopy(net_w))
        self.optimizer.load_state_dict(copy.deepcopy(op_dict))
        
        self.model.train()
        
        epoch_losses = []
        for epoch in range(self.args.sup_local_ep):
            batch_losses = []
            for _, image_batch, label_batch in dataloader:
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                
                # Ensure labels are in the correct format (long tensor)
                label_batch = label_batch.long().squeeze()

                # Forward pass
                _, _, outputs = self.model(image_batch)

                # Calculate loss
                loss = self.loss_fn(outputs, label_batch)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()

                batch_losses.append(loss.item())
            
            epoch_losses.append(np.mean(batch_losses))

        # Return the updated model state, average loss, and optimizer state
        return (
            self.model.state_dict(),
            np.mean(epoch_losses),
            copy.deepcopy(self.optimizer.state_dict())
        )