"""CCNN with the DEAP Dataset
======================================
In this case, we introduce how to use TorchEEG to train a Continuous Convolutional Neural Network (CCNN) on the DEAP dataset for emotion classification.
"""

import logging
import os
import random
import time

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torcheeg import transforms
from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.constants.emotion_recognition.deap import \
    DEAP_CHANNEL_LOCATION_DICT
from torcheeg.model_selection import KFoldGroupbyTrial
from torcheeg.models import CCNN
from torcheeg.trainers import ClassificationTrainer

logger = logging.getLogger('CCNN with the DEAP Dataset')
logger.setLevel(logging.DEBUG)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)



class MyClassificationTrainer(ClassificationTrainer):
    def log(self, *args, **kwargs):
        if self.is_main:
            logger.info(*args, **kwargs)



def run(dataset):

    k_fold = KFoldGroupbyTrial(n_splits=5,
                            split_path=f'{os.getcwd()}/split')

    ######################################################################
    # Step 3: Define the Model and Start Training
    #
    # We first use a loop to get the dataset in each cross-validation. In each cross-validation, we initialize the CCNN model and define the hyperparameters. For example, each EEG sample contains 4-channel features from 4 sub-bands, the grid size is 9 times 9, etc.
    #
    # We then initialize the trainer and set the hyperparameters in the trained model, such as the learning rate, the equipment used, etc. The :obj:`fit` method receives the training dataset and starts training the model. The :obj:`test` method receives a test dataset and reports the test results. The :obj:`save_state_dict` method can save the trained model.

    for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
        # Initialize the model
        model = CCNN(in_channels=4, grid_size=(9, 9), num_classes=2, dropout=0.5)

        # Initialize the trainer and use the 0-th GPU for training, or set device_ids=[] to use CPU
        trainer = MyClassificationTrainer(model=model,
                                        lr=1e-4,
                                        weight_decay=1e-4,
                                        device_ids=[0])

        # Initialize several batches of training samples and test samples
        train_loader = DataLoader(train_dataset,
                                batch_size=256,
                                shuffle=True,
                                num_workers=4)
        val_loader = DataLoader(val_dataset,
                                batch_size=256,
                                shuffle=False,
                                num_workers=4)

        # Do 50 rounds of training
        trainer.fit(train_loader, val_loader, num_epochs=50)
        trainer.test(val_loader)
        trainer.save_state_dict(f'{os.getcwd()}/examples_deap_ccnn/weight/{i}.pth')