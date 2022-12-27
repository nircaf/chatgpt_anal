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
# from autoPyTorch.api.tabular_classification import TabularClassificationTask

# data and metric imports
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import pandas as pd
import h2o
from h2o.automl import H2OAutoML
from tpot import TPOTClassifier

def tpot_train(X_train, X_test, y_train, y_test):
    tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))
    # save model

    tpot.export(f"{os.getcwd()}/saved_models/tpot_digits_pipeline.py")


def h2o_train(X_train, X_test, y_train, y_test):
    # Start the H2O cluster (locally)
    h2o.init()
    # change y train and test 0 to 2
    y_train[y_train == 0] = 2
    y_test[y_test == 0] = 2
    # concat x and y train to one df
    train = pd.concat([X_train, y_train], axis=1)
    # concat x and y test to one df
    test = pd.concat([X_test, y_test], axis=1)
    # rename last column to y
    train.rename(columns={train.columns[-1]: 'y'}, inplace=True)
    test.rename(columns={test.columns[-1]: 'y'}, inplace=True)
    # change train and test to h2o.H2OFrame
    train = h2o.H2OFrame(train)
    test = h2o.H2OFrame(test)
    x = train.columns[:-1]
    y = train.columns[-1]
    # For binary classification, response should be a factor
    train[y] = train[y].asfactor()
    test[y] = test[y].asfactor()
    # Run AutoML for 20 base models
    aml = H2OAutoML(max_models=20, seed=1)
    aml.train( y=y,x=x, training_frame=train)
    a = aml.leader.params.keys()
    # View the AutoML Leaderboard
    lb = aml.leaderboard
    print(lb.head(rows=lb.nrows))  # Print all rows instead of default (10 rows)
    # To generate predictions on a test set, you can make predictions
    # directly on the `H2OAutoML` object or on the leader model
    # object directly
    preds = aml.predict(test)
    # Get leaderboard with all possible columns
    lb = h2o.automl.get_leaderboard(aml, extra_columns = "ALL")
    print(lb)
    # Get the best model using the metric
    m = aml.leader
    # this is equivalent to
    m = aml.get_best_model()
    print(m)
    # save the model in saved_models folder
    h2o.save_model(model=m, path=f"{os.getcwd()}/saved_models", force=True)

def autoPyTorch_run(X_train, X_test, y_train, y_test):
    # initialise Auto-PyTorch api
    api = TabularClassificationTask()

    # Search for an ensemble of machine learning algorithms
    api.search(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        optimize_metric='accuracy',
        total_walltime_limit=300,
        func_eval_time_limit_secs=50
    )

    # Calculate test accuracy
    y_pred = api.predict(X_test)
    score = api.score(y_pred, y_test)
    print("Accuracy score", score)
    # Print statistics from search
    print(api.sprint_statistics())
    # Print the final ensemble built by AutoPyTorch
    print(api.show_models())

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
