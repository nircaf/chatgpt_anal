import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
def tpot_train(training_features, testing_features, training_target, testing_target):
    # Average CV score on the training set was: 0.9524362697644401
    exported_pipeline = make_pipeline(
        StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.4, min_samples_leaf=3, min_samples_split=6, n_estimators=100)),
        GradientBoostingClassifier(learning_rate=1.0, max_depth=4, max_features=0.7500000000000001, min_samples_leaf=7, min_samples_split=16, n_estimators=100, subsample=0.8)
    )
    # Fix random state for all the steps in exported pipeline
    set_param_recursive(exported_pipeline.steps, 'random_state', 42)

    exported_pipeline.fit(training_features, training_target)
    results = exported_pipeline.predict(testing_features)
    print(f'accuracy: {np.sum(results == testing_target.T.to_numpy()) / len(testing_target)}')
    return exported_pipeline
