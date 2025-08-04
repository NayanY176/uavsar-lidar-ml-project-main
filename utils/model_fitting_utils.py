# Third Code
import pandas as pd
from typing import Dict
from IPython.display import display
import numpy as np # Added for consistent numerical operations, especially for MBE

from utils.model_utils import compute_metrics
from utils.pytorch_model import RegressionNN
from utils.pytorch_training import train, predict
from utils.pytorch_dataset import create_dataset_for_dnn

import xgboost as xgb
from sklearn import set_config
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb # New: Import LightGBM

set_config(
    transform_output="pandas"
)

import torch
import torch.nn as nn
import torch.optim as optim

xgb_device = ("cuda" if torch.cuda.is_available() else "cpu")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


initial_params = {
    'extra_trees': {
        'n_estimators': 100,
        'max_depth': None,
        'criterion': 'squared_error',
        'n_jobs': -1,
        'random_state': 42
    },
    'xgboost': {
        "sampling_method": "gradient_based", # <-- This is now supported
        'objective': 'reg:squarederror',
        "min_child_weight": 30,
        'learning_rate': 0.05,
        'tree_method': 'gpu_hist',          # <-- Changed to use the GPU
        'booster': 'gbtree',
        'device': 'cuda',                   # <-- Explicitly set for GPU
        'max_depth': 0,
        "subsample": 1,
        "max_bin": 5096,
        "trees": 1000,
        "seed": 42
    },
    'pytorch_nn': {
        'hidden_size1': 2048,
        'hidden_size2': 1500,
        'hidden_size3': 1000,
        'num_epochs': 15,
        'batch_size': 128,
        'learning_rate': 0.0001,
        'verbose': True
    },
    'linear_regression': {
        'fit_intercept': True,
        'n_jobs': -1
    },
    'random_forest': {
        'n_estimators': 100, # Changed from 1 to 100 as 1 tree is rarely useful for Random Forest
        'max_depth': None,
        'criterion': 'squared_error',
        'n_jobs': -1,
        'random_state': 42
    },
    'lightgbm': { # New: Default parameters for LightGBM Regressor
        'objective': 'regression_l1', # 'regression_l1' for MAE, 'regression_l2' for MSE
        'metric': 'rmse',
        'n_estimators': 50,
        'learning_rate': 0.05,
        'num_leaves': 31, # Default is 31, can be tuned
        'max_depth': -1, # -1 means no limit
        'min_child_samples': 20,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1 # Suppress verbose output during training
    }
}

class ModelFitting:

    def __init__(
        self,
        var: str,
        split: Dict[str, pd.DataFrame],
        model_name: str,
        **model_params
    ):
        self.var = var
        self.split = split
        self.model_name = model_name

        if self.model_name == 'extra_trees':
            self.model_params = {**initial_params['extra_trees'], **model_params}

        elif self.model_name == 'xgboost':
            self.model_params = {**initial_params['xgboost'], **model_params}

        elif self.model_name == 'pytorch_nn':
            self.model_params = {**initial_params['pytorch_nn'], **model_params}

        elif self.model_name == 'linear_regression':
            self.model_params = {**initial_params['linear_regression'], **model_params}

        elif self.model_name == 'random_forest':
            self.model_params = {**initial_params['random_forest'], **model_params}

        elif self.model_name == 'lightgbm': # New: Handle lightgbm
            self.model_params = {**initial_params['lightgbm'], **model_params}

        else:
            raise ValueError(f'Invalid model name: {self.model_name}. \nPlease choose from: extra_trees, xgboost, pytorch_nn, linear_regression, random_forest, lightgbm.')


    def fit_model(self) -> None:
        """
        A function that fits the model to the training data.
        """

        if self.model_name == 'extra_trees':

            self.model = ExtraTreesRegressor(**self.model_params)
            X_train, y_train = self.split['X_temp'][self.var], self.split['y_temp']
            self.model.fit(X_train, y_train)

        elif self.model_name == 'xgboost':

            dtrain=xgb.DMatrix(self.split['X_temp'][self.var], label=self.split['y_temp'])
            n_trees = self.model_params["trees"]
            boosting_params = self.model_params.copy()
            boosting_params.pop("trees") # 'trees' is for num_boost_round, not a direct param to xgb.train

            print("Parameters being used for xgb.train:", boosting_params)

            self.model = xgb.train(
                params=boosting_params,
                dtrain=dtrain,
                num_boost_round=n_trees
            )

        elif self.model_name == 'pytorch_nn':

            self.loader = create_dataset_for_dnn(
                split=self.split,
                columns_of_interest=self.var,
                batch_size=self.model_params['batch_size']
            )

            input_size = self.loader['train_dataloader'].dataset.features.shape[1]
            hidden_size1 = self.model_params['hidden_size1']
            hidden_size2 = self.model_params['hidden_size2']
            hidden_size3 = self.model_params['hidden_size3']

            self.model = RegressionNN(
                input_size=input_size,
                hidden_size1=hidden_size1,
                hidden_size2=hidden_size2,
                hidden_size3=hidden_size3
            )

            optimizer = optim.Adam(self.model.parameters(), lr= self.model_params['learning_rate'])
            criterion = nn.MSELoss()

            self.history = train(
                model=self.model,
                train_loader=self.loader['train_dataloader'],
                val_loader=self.loader['val_dataloader'],
                epochs= self.model_params['num_epochs'],
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                metric='mae',
                verbose=self.model_params['verbose']
            )

        elif self.model_name == 'linear_regression':
            self.model = LinearRegression(**self.model_params)
            X_train, y_train = self.split['X_temp'][self.var], self.split['y_temp']
            self.model.fit(X_train, y_train)

        elif self.model_name == 'random_forest':
            self.model = RandomForestRegressor(**self.model_params)
            X_train, y_train = self.split['X_temp'][self.var], self.split['y_temp']
            self.model.fit(X_train, y_train)

        elif self.model_name == 'lightgbm': # New: Fit LightGBM Regressor
            self.model = lgb.LGBMRegressor(**self.model_params)
            X_train, y_train = self.split['X_temp'][self.var], self.split['y_temp']
            self.model.fit(X_train, y_train)

        else:
            raise ValueError(f'Invalid model name: {self.model_name}.')

    def make_predictions(self) -> Dict[str, pd.DataFrame]:
        """
        A function that generates predictions for the model.
        """

        if self.model_name == 'extra_trees':

            self.y_pred_test = self.model.predict(self.split['X_test'][self.var])
            y_pred_test_df = pd.DataFrame(
                data = self.y_pred_test,
                columns = ['snow_depth_pred']
            )

            self.y_pred_train = self.model.predict(self.split['X_temp'][self.var])
            y_pred_train_df = pd.DataFrame(
                data = self.y_pred_train,
                columns = ['snow_depth_pred']
            )

            return {
                'y_pred_test': y_pred_test_df,
                'y_pred_train': y_pred_train_df
            }


        elif self.model_name == 'xgboost':

            self.y_pred_test = self.model.predict(
                xgb.DMatrix(self.split['X_test'][self.var])
            )
            y_pred_test_df = pd.DataFrame(
                data = self.y_pred_test,
                columns = ['snow_depth_pred']
            )

            self.y_pred_train = self.model.predict(
                xgb.DMatrix(self.split['X_temp'][self.var])
            )
            y_pred_train_df = pd.DataFrame(
                data = self.y_pred_train,
                columns = ['snow_depth_pred']
            )

            return {
                'y_pred_test': y_pred_test_df,
                'y_pred_train': y_pred_train_df
            }

        elif self.model_name == 'pytorch_nn':

            self.predictions_test = predict(
                model=self.model,
                test_loader=self.loader['test_dataloader'],
                device=device
            )

            self.predictions_train = predict(
                model=self.model,
                test_loader=self.loader['train_dataloader'],
                device=device
            )

            y_pred_test_df = pd.DataFrame(
                data = self.predictions_test['predictions'],
                columns = ['snow_depth_pred']
            )

            y_pred_train_df = pd.DataFrame(
                data = self.predictions_train['predictions'],
                columns = ['snow_depth_pred']
            )

            return {
                'y_pred_test': y_pred_test_df,
                'y_pred_train': y_pred_train_df
            }

        elif self.model_name == 'linear_regression':
            self.y_pred_test = self.model.predict(self.split['X_test'][self.var])
            y_pred_test_df = pd.DataFrame(
                data = self.y_pred_test,
                columns = ['snow_depth_pred']
            )

            self.y_pred_train = self.model.predict(self.split['X_temp'][self.var])
            y_pred_train_df = pd.DataFrame(
                data = self.y_pred_train,
                columns = ['snow_depth_pred']
            )

            return {
                'y_pred_test': y_pred_test_df,
                'y_pred_train': y_pred_train_df
            }

        elif self.model_name == 'random_forest':
            self.y_pred_test = self.model.predict(self.split['X_test'][self.var])
            y_pred_test_df = pd.DataFrame(
                data = self.y_pred_test,
                columns = ['snow_depth_pred']
            )

            self.y_pred_train = self.model.predict(self.split['X_temp'][self.var])
            y_pred_train_df = pd.DataFrame(
                data = self.y_pred_train,
                columns = ['snow_depth_pred']
            )

            return {
                'y_pred_test': y_pred_test_df,
                'y_pred_train': y_pred_train_df
            }

        elif self.model_name == 'lightgbm': # New: Make LightGBM predictions
            self.y_pred_test = self.model.predict(self.split['X_test'][self.var])
            y_pred_test_df = pd.DataFrame(
                data = self.y_pred_test,
                columns = ['snow_depth_pred']
            )

            self.y_pred_train = self.model.predict(self.split['X_temp'][self.var])
            y_pred_train_df = pd.DataFrame(
                data = self.y_pred_train,
                columns = ['snow_depth_pred']
            )

            return {
                'y_pred_test': y_pred_test_df,
                'y_pred_train': y_pred_train_df
            }

        else:
            raise ValueError(f'Invalid model name: {self.model_name}. \nPlease choose from: extra_trees, xgboost, pytorch_nn, linear_regression, random_forest, lightgbm.')


    def evaluate_model(self) -> pd.DataFrame:

        """
        A function that evaluates the performance of a model using the RMSE, MBE and R2 metrics.
        """

        if self.model_name == 'extra_trees':

            train_eval=compute_metrics(
                y_true=self.split['y_temp'],
                y_pred=self.y_pred_train,
                model_name=self.model_name + '_train'
            )

            test_eval=compute_metrics(
                y_true=self.split['y_test'],
                y_pred=self.y_pred_test,
                model_name=self.model_name + '_test'
            )

            eval_df = pd.concat([train_eval, test_eval], axis=1)
            display(eval_df)

            return eval_df

        elif self.model_name == 'xgboost':

            train_eval=compute_metrics(
                y_true=self.split['y_temp'],
                y_pred=self.y_pred_train,
                model_name=self.model_name + '_train'
            )

            test_eval=compute_metrics(
                y_true=self.split['y_test'],
                y_pred=self.y_pred_test,
                model_name=self.model_name + '_test'
            )

            eval_df = pd.concat([train_eval, test_eval], axis=1)
            display(eval_df)

            return eval_df

        elif self.model_name == 'pytorch_nn':

            train_eval=compute_metrics(
                y_true=self.predictions_train['labels'],
                y_pred=self.predictions_train['predictions'],
                model_name=self.model_name + '_train'
            )

            test_eval=compute_metrics(
                y_true=self.predictions_test['labels'],
                y_pred=self.predictions_test['predictions'],
                model_name=self.model_name + '_test'
            )

            eval_df = pd.concat([train_eval, test_eval], axis=1)
            display(eval_df)

            return eval_df

        elif self.model_name == 'linear_regression':
            train_eval=compute_metrics(
                y_true=self.split['y_temp'],
                y_pred=self.y_pred_train,
                model_name=self.model_name + '_train'
            )

            test_eval=compute_metrics(
                y_true=self.split['y_test'],
                y_pred=self.y_pred_test,
                model_name=self.model_name + '_test'
            )

            eval_df = pd.concat([train_eval, test_eval], axis=1)
            display(eval_df)

            return eval_df

        elif self.model_name == 'random_forest':
            train_eval=compute_metrics(
                y_true=self.split['y_temp'],
                y_pred=self.y_pred_train,
                model_name=self.model_name + '_train'
            )

            test_eval=compute_metrics(
                y_true=self.split['y_test'],
                y_pred=self.y_pred_test,
                model_name=self.model_name + '_test'
            )

            eval_df = pd.concat([train_eval, test_eval], axis=1)
            display(eval_df)

            return eval_df

        elif self.model_name == 'lightgbm': # New: Evaluate LightGBM
            train_eval=compute_metrics(
                y_true=self.split['y_temp'],
                y_pred=self.y_pred_train,
                model_name=self.model_name + '_train'
            )

            test_eval=compute_metrics(
                y_true=self.split['y_test'],
                y_pred=self.y_pred_test,
                model_name=self.model_name + '_test'
            )

            eval_df = pd.concat([train_eval, test_eval], axis=1)
            display(eval_df)

            return eval_df

        else:
            raise ValueError(f'Invalid model name: {self.model_name}. \nPlease choose from: extra_trees, xgboost, pytorch_nn, linear_regression, random_forest, lightgbm.')


    def get_importance(self) -> pd.DataFrame:

        if self.model_name == 'extra_trees':

            feature_importance = pd.DataFrame(
                data = {
                    'feature': self.split['X_temp'][self.var].columns,
                    'importance': self.model.feature_importances_
                }
            ).sort_values(by='importance', ascending=False)

            display(feature_importance)

            return feature_importance

        elif self.model_name == 'xgboost':

            xgb_importance = self.model.get_score(importance_type='gain')
            feature_importance_list = []
            for col in self.split['X_temp'][self.var].columns:
                feature_importance_list.append({'feature': col, 'importance': xgb_importance.get(col, 0)})

            feature_importance = pd.DataFrame(feature_importance_list).sort_values(by='importance', ascending=False)

            display(feature_importance)

            return feature_importance

        elif self.model_name == 'pytorch_nn':
            print('No direct feature importance for PyTorch NNs in this setup.')
            return pd.DataFrame()

        elif self.model_name == 'linear_regression':
            feature_importance = pd.DataFrame(
                data = {
                    'feature': self.split['X_temp'][self.var].columns,
                    'importance': self.model.coef_
                }
            ).sort_values(by='importance', ascending=False)

            display(feature_importance)

            return feature_importance

        elif self.model_name == 'random_forest':
            feature_importance = pd.DataFrame(
                data = {
                    'feature': self.split['X_temp'][self.var].columns,
                    'importance': self.model.feature_importances_
                }
            ).sort_values(by='importance', ascending=False)

            display(feature_importance)

            return feature_importance

        elif self.model_name == 'lightgbm': # New: Get feature importance for LightGBM
            feature_importance = pd.DataFrame(
                data = {
                    'feature': self.split['X_temp'][self.var].columns,
                    'importance': self.model.feature_importances_
                }
            ).sort_values(by='importance', ascending=False)

            display(feature_importance)

            return feature_importance

        else:
            raise ValueError(f'Invalid model name: {self.model_name}.')
### Second Code
# import pandas as pd
# from typing import Dict
# from IPython.display import display
# import numpy as np # Added for consistent numerical operations, especially for MBE

# from utils.model_utils import compute_metrics
# from utils.pytorch_model import RegressionNN
# from utils.pytorch_training import train, predict
# from utils.pytorch_dataset import create_dataset_for_dnn

# import xgboost as xgb
# from sklearn import set_config
# from sklearn.ensemble import ExtraTreesRegressor
# from sklearn.linear_model import LinearRegression # New: Import LinearRegression

# set_config(
#     transform_output="pandas"
# )

# import torch
# import torch.nn as nn
# import torch.optim as optim

# xgb_device = ("cuda" if torch.cuda.is_available() else "cpu")

# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )


# initial_params = {
#     'extra_trees': {
#         'n_estimators': 100,
#         'max_depth': None,
#         'criterion': 'squared_error',
#         'n_jobs': -1,
#         'random_state': 42
#     },
#     'xgboost': {
#         "sampling_method": "gradient_based",
#         'objective': 'reg:squarederror',
#         "min_child_weight": 30,
#         'learning_rate': 0.05,
#         'tree_method': 'hist',
#         'booster': 'gbtree',
#         'device': xgb_device,
#         'max_depth': 0,
#         "subsample": 1,
#         "max_bin":5096,
#         "trees": 1000,
#         "seed": 42
#     },
#     'pytorch_nn': {
#         'hidden_size1': 2048,
#         'hidden_size2': 1500,
#         'hidden_size3': 1000,
#         'num_epochs': 15,
#         'batch_size': 128,
#         'learning_rate': 0.0001,
#         'verbose': True
#     },
#     'linear_regression': { # New: Default parameters for Linear Regression
#         'fit_intercept': True,
#         'n_jobs': -1
#     }
# }

# class ModelFitting:

#     def __init__(
#         self,
#         var: str,
#         split: Dict[str, pd.DataFrame],
#         model_name: str,
#         **model_params
#     ):
#         self.var = var
#         self.split = split
#         self.model_name = model_name

#         if self.model_name == 'extra_trees':
#             self.model_params = {**initial_params['extra_trees'], **model_params}

#         elif self.model_name == 'xgboost':
#             self.model_params = {**initial_params['xgboost'], **model_params}

#         elif self.model_name == 'pytorch_nn':
#             self.model_params = {**initial_params['pytorch_nn'], **model_params}

#         elif self.model_name == 'linear_regression': # New: Handle linear_regression
#             self.model_params = {**initial_params['linear_regression'], **model_params}

#         else:
#             raise ValueError(f'Invalid model name: {self.model_name}. \nPlease choose from: extra_trees, xgboost, pytorch_nn, linear_regression.')


#     def fit_model(self) -> None:
#         """
#         A function that fits the model to the training data.
#         """

#         if self.model_name == 'extra_trees':

#             self.model = ExtraTreesRegressor(**self.model_params)
#             X_train, y_train = self.split['X_temp'][self.var], self.split['y_temp']
#             self.model.fit(X_train, y_train)

#         elif self.model_name == 'xgboost':

#             dtrain=xgb.DMatrix(self.split['X_temp'][self.var], label=self.split['y_temp'])
#             n_trees = self.model_params["trees"]
#             boosting_params = self.model_params.copy()
#             boosting_params.pop("trees")

#             self.model = xgb.train(
#                 params=boosting_params,
#                 dtrain=dtrain,
#                 num_boost_round=n_trees
#             )

#         elif self.model_name == 'pytorch_nn':

#             self.loader = create_dataset_for_dnn(
#                 split=self.split,
#                 columns_of_interest=self.var,
#                 batch_size=self.model_params['batch_size']
#             )

#             input_size = self.loader['train_dataloader'].dataset.features.shape[1]
#             hidden_size1 = self.model_params['hidden_size1']
#             hidden_size2 = self.model_params['hidden_size2']
#             hidden_size3 = self.model_params['hidden_size3']

#             self.model = RegressionNN(
#                 input_size=input_size,
#                 hidden_size1=hidden_size1,
#                 hidden_size2=hidden_size2,
#                 hidden_size3=hidden_size3
#             )

#             optimizer = optim.Adam(self.model.parameters(), lr= self.model_params['learning_rate'])
#             criterion = nn.MSELoss()

#             self.history = train(
#                 model=self.model,
#                 train_loader=self.loader['train_dataloader'],
#                 val_loader=self.loader['val_dataloader'],
#                 epochs= self.model_params['num_epochs'],
#                 criterion=criterion,
#                 optimizer=optimizer,
#                 device=device,
#                 metric='mae',
#                 verbose=self.model_params['verbose']
#             )
        
#         elif self.model_name == 'linear_regression': # New: Fit Linear Regression
#             self.model = LinearRegression(**self.model_params)
#             X_train, y_train = self.split['X_temp'][self.var], self.split['y_temp']
#             self.model.fit(X_train, y_train)

#         else:
#             raise ValueError(f'Invalid model name: {self.model_name}.')

#     def make_predictions(self) -> Dict[str, pd.DataFrame]:
#         """
#         A function that generates predictions for the model.
#         """

#         if self.model_name == 'extra_trees':

#             self.y_pred_test = self.model.predict(self.split['X_test'][self.var])
#             y_pred_test_df = pd.DataFrame(
#                 data = self.y_pred_test,
#                 columns = ['snow_depth_pred']
#             )

#             self.y_pred_train = self.model.predict(self.split['X_temp'][self.var])
#             y_pred_train_df = pd.DataFrame(
#                 data = self.y_pred_train,
#                 columns = ['snow_depth_pred']
#             )

#             return {
#                 'y_pred_test': y_pred_test_df,
#                 'y_pred_train': y_pred_train_df
#             }


#         elif self.model_name == 'xgboost':

#             self.y_pred_test = self.model.predict(
#                 xgb.DMatrix(self.split['X_test'][self.var])
#             )
#             y_pred_test_df = pd.DataFrame(
#                 data = self.y_pred_test,
#                 columns = ['snow_depth_pred']
#             )

#             self.y_pred_train = self.model.predict(
#                 xgb.DMatrix(self.split['X_temp'][self.var])
#             )
#             y_pred_train_df = pd.DataFrame(
#                 data = self.y_pred_train,
#                 columns = ['snow_depth_pred']
#             )

#             return {
#                 'y_pred_test': y_pred_test_df,
#                 'y_pred_train': y_pred_train_df
#             }

#         elif self.model_name == 'pytorch_nn':

#             self.predictions_test = predict(
#                 model=self.model,
#                 test_loader=self.loader['test_dataloader'],
#                 device=device
#             )

#             self.predictions_train = predict(
#                 model=self.model,
#                 test_loader=self.loader['train_dataloader'],
#                 device=device
#             )

#             y_pred_test_df = pd.DataFrame(
#                 data = self.predictions_test['predictions'],
#                 columns = ['snow_depth_pred']
#             )

#             y_pred_train_df = pd.DataFrame(
#                 data = self.predictions_train['predictions'],
#                 columns = ['snow_depth_pred']
#             )

#             return {
#                 'y_pred_test': y_pred_test_df,
#                 'y_pred_train': y_pred_train_df
#             }

#         elif self.model_name == 'linear_regression': # New: Make Linear Regression predictions
#             self.y_pred_test = self.model.predict(self.split['X_test'][self.var])
#             y_pred_test_df = pd.DataFrame(
#                 data = self.y_pred_test,
#                 columns = ['snow_depth_pred']
#             )

#             self.y_pred_train = self.model.predict(self.split['X_temp'][self.var])
#             y_pred_train_df = pd.DataFrame(
#                 data = self.y_pred_train,
#                 columns = ['snow_depth_pred']
#             )

#             return {
#                 'y_pred_test': y_pred_test_df,
#                 'y_pred_train': y_pred_train_df
#             }

#         else:
#             raise ValueError(f'Invalid model name: {self.model_name}. \nPlease choose from: extra_trees, xgboost, pytorch_nn, linear_regression.')


#     def evaluate_model(self) -> pd.DataFrame:

#         """
#         A function that evaluates the performance of a model using the RMSE, MBE and R2 metrics.
#         """

#         if self.model_name == 'extra_trees':

#             train_eval=compute_metrics(
#                 y_true=self.split['y_temp'],
#                 y_pred=self.y_pred_train,
#                 model_name=self.model_name + '_train'
#             )

#             test_eval=compute_metrics(
#                 y_true=self.split['y_test'],
#                 y_pred=self.y_pred_test,
#                 model_name=self.model_name + '_test'
#             )

#             eval_df = pd.concat([train_eval, test_eval], axis=1)
#             display(eval_df)

#             return eval_df

#         elif self.model_name == 'xgboost':

#             train_eval=compute_metrics(
#                 y_true=self.split['y_temp'],
#                 y_pred=self.y_pred_train,
#                 model_name=self.model_name + '_train'
#             )

#             test_eval=compute_metrics(
#                 y_true=self.split['y_test'],
#                 y_pred=self.y_pred_test,
#                 model_name=self.model_name + '_test'
#             )

#             eval_df = pd.concat([train_eval, test_eval], axis=1)
#             display(eval_df)

#             return eval_df

#         elif self.model_name == 'pytorch_nn':

#             train_eval=compute_metrics(
#                 y_true=self.predictions_train['labels'],
#                 y_pred=self.predictions_train['predictions'],
#                 model_name=self.model_name + '_train'
#             )

#             test_eval=compute_metrics(
#                 y_true=self.predictions_test['labels'],
#                 y_pred=self.predictions_test['predictions'],
#                 model_name=self.model_name + '_test'
#             )

#             eval_df = pd.concat([train_eval, test_eval], axis=1)
#             display(eval_df)

#             return eval_df

#         elif self.model_name == 'linear_regression': # New: Evaluate Linear Regression
#             train_eval=compute_metrics(
#                 y_true=self.split['y_temp'],
#                 y_pred=self.y_pred_train,
#                 model_name=self.model_name + '_train'
#             )

#             test_eval=compute_metrics(
#                 y_true=self.split['y_test'],
#                 y_pred=self.y_pred_test,
#                 model_name=self.model_name + '_test'
#             )

#             eval_df = pd.concat([train_eval, test_eval], axis=1)
#             display(eval_df)

#             return eval_df

#         else:
#             raise ValueError(f'Invalid model name: {self.model_name}. \nPlease choose from: extra_trees, xgboost, pytorch_nn, linear_regression.')


#     def get_importance(self) -> pd.DataFrame:

#         if self.model_name == 'extra_trees':

#             feature_importance = pd.DataFrame(
#                 data = {
#                     'feature': self.split['X_temp'][self.var].columns,
#                     'importance': self.model.feature_importances_
#                 }
#             ).sort_values(by='importance', ascending=False)

#             display(feature_importance)

#             return feature_importance

#         elif self.model_name == 'xgboost':

#             # XGBoost get_score returns a dictionary where keys are feature names.
#             # Ensure the order matches self.split['X_temp'][self.var].columns
#             # or handle missing features in the score.
#             xgb_importance = self.model.get_score(importance_type='gain')
#             feature_importance_list = []
#             for col in self.split['X_temp'][self.var].columns:
#                 feature_importance_list.append({'feature': col, 'importance': xgb_importance.get(col, 0)}) # Get 0 if feature not found

#             feature_importance = pd.DataFrame(feature_importance_list).sort_values(by='importance', ascending=False)


#             display(feature_importance)

#             return feature_importance

#         elif self.model_name == 'pytorch_nn':
#             print('No direct feature importance for PyTorch NNs in this setup.')
#             return pd.DataFrame() # Return empty DataFrame

#         elif self.model_name == 'linear_regression': # New: Get feature importance for Linear Regression
#             feature_importance = pd.DataFrame(
#                 data = {
#                     'feature': self.split['X_temp'][self.var].columns,
#                     'importance': self.model.coef_
#                 }
#             ).sort_values(by='importance', ascending=False) # Sort by absolute importance if you prefer

#             display(feature_importance)

#             return feature_importance

#         else:
#             raise ValueError(f'Invalid model name: {self.model_name}.')

# Third Code
# import pandas as pd
# from typing import Dict
# from IPython.display import display
# import numpy as np # Added for consistent numerical operations, especially for MBE

# from utils.model_utils import compute_metrics
# from utils.pytorch_model import RegressionNN
# from utils.pytorch_training import train, predict
# from utils.pytorch_dataset import create_dataset_for_dnn

# import xgboost as xgb
# from sklearn import set_config
# from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor # New: Import RandomForestRegressor
# from sklearn.linear_model import LinearRegression

# set_config(
#     transform_output="pandas"
# )

# import torch
# import torch.nn as nn
# import torch.optim as optim

# xgb_device = ("cuda" if torch.cuda.is_available() else "cpu")

# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )


# initial_params = {
#     'extra_trees': {
#         'n_estimators': 100,
#         'max_depth': None,
#         'criterion': 'squared_error',
#         'n_jobs': -1,
#         'random_state': 42
#     },
#     'xgboost': {
#         "sampling_method": "gradient_based",
#         'objective': 'reg:squarederror',
#         "min_child_weight": 30,
#         'learning_rate': 0.05,
#         'tree_method': 'hist',
#         'booster': 'gbtree',
#         'device': xgb_device,
#         'max_depth': 0,
#         "subsample": 1,
#         "max_bin":5096,
#         "trees": 1000,
#         "seed": 42
#     },
#     'pytorch_nn': {
#         'hidden_size1': 2048,
#         'hidden_size2': 1500,
#         'hidden_size3': 1000,
#         'num_epochs': 15,
#         'batch_size': 128,
#         'learning_rate': 0.0001,
#         'verbose': True
#     },
#     'linear_regression': {
#         'fit_intercept': True,
#         'n_jobs': -1
#     },
#     'random_forest': { # New: Default parameters for Random Forest Regressor
#         'n_estimators': 1,
#         'max_depth': None,
#         'criterion': 'squared_error',
#         'n_jobs': -1,
#         'random_state': 42
#     }
# }

# class ModelFitting:

#     def __init__(
#         self,
#         var: str,
#         split: Dict[str, pd.DataFrame],
#         model_name: str,
#         **model_params
#     ):
#         self.var = var
#         self.split = split
#         self.model_name = model_name

#         if self.model_name == 'extra_trees':
#             self.model_params = {**initial_params['extra_trees'], **model_params}

#         elif self.model_name == 'xgboost':
#             self.model_params = {**initial_params['xgboost'], **model_params}

#         elif self.model_name == 'pytorch_nn':
#             self.model_params = {**initial_params['pytorch_nn'], **model_params}

#         elif self.model_name == 'linear_regression':
#             self.model_params = {**initial_params['linear_regression'], **model_params}

#         elif self.model_name == 'random_forest': # New: Handle random_forest
#             self.model_params = {**initial_params['random_forest'], **model_params}

#         else:
#             raise ValueError(f'Invalid model name: {self.model_name}. \nPlease choose from: extra_trees, xgboost, pytorch_nn, linear_regression, random_forest.')


#     def fit_model(self) -> None:
#         """
#         A function that fits the model to the training data.
#         """

#         if self.model_name == 'extra_trees':

#             self.model = ExtraTreesRegressor(**self.model_params)
#             X_train, y_train = self.split['X_temp'][self.var], self.split['y_temp']
#             self.model.fit(X_train, y_train)

#         elif self.model_name == 'xgboost':

#             dtrain=xgb.DMatrix(self.split['X_temp'][self.var], label=self.split['y_temp'])
#             n_trees = self.model_params["trees"]
#             boosting_params = self.model_params.copy()
#             boosting_params.pop("trees")

#             self.model = xgb.train(
#                 params=boosting_params,
#                 dtrain=dtrain,
#                 num_boost_round=n_trees
#             )

#         elif self.model_name == 'pytorch_nn':

#             self.loader = create_dataset_for_dnn(
#                 split=self.split,
#                 columns_of_interest=self.var,
#                 batch_size=self.model_params['batch_size']
#             )

#             input_size = self.loader['train_dataloader'].dataset.features.shape[1]
#             hidden_size1 = self.model_params['hidden_size1']
#             hidden_size2 = self.model_params['hidden_size2']
#             hidden_size3 = self.model_params['hidden_size3']

#             self.model = RegressionNN(
#                 input_size=input_size,
#                 hidden_size1=hidden_size1,
#                 hidden_size2=hidden_size2,
#                 hidden_size3=hidden_size3
#             )

#             optimizer = optim.Adam(self.model.parameters(), lr= self.model_params['learning_rate'])
#             criterion = nn.MSELoss()

#             self.history = train(
#                 model=self.model,
#                 train_loader=self.loader['train_dataloader'],
#                 val_loader=self.loader['val_dataloader'],
#                 epochs= self.model_params['num_epochs'],
#                 criterion=criterion,
#                 optimizer=optimizer,
#                 device=device,
#                 metric='mae',
#                 verbose=self.model_params['verbose']
#             )

#         elif self.model_name == 'linear_regression':
#             self.model = LinearRegression(**self.model_params)
#             X_train, y_train = self.split['X_temp'][self.var], self.split['y_temp']
#             self.model.fit(X_train, y_train)

#         elif self.model_name == 'random_forest': # New: Fit Random Forest Regressor
#             self.model = RandomForestRegressor(**self.model_params)
#             X_train, y_train = self.split['X_temp'][self.var], self.split['y_temp']
#             self.model.fit(X_train, y_train)

#         else:
#             raise ValueError(f'Invalid model name: {self.model_name}.')

#     def make_predictions(self) -> Dict[str, pd.DataFrame]:
#         """
#         A function that generates predictions for the model.
#         """

#         if self.model_name == 'extra_trees':

#             self.y_pred_test = self.model.predict(self.split['X_test'][self.var])
#             y_pred_test_df = pd.DataFrame(
#                 data = self.y_pred_test,
#                 columns = ['snow_depth_pred']
#             )

#             self.y_pred_train = self.model.predict(self.split['X_temp'][self.var])
#             y_pred_train_df = pd.DataFrame(
#                 data = self.y_pred_train,
#                 columns = ['snow_depth_pred']
#             )

#             return {
#                 'y_pred_test': y_pred_test_df,
#                 'y_pred_train': y_pred_train_df
#             }


#         elif self.model_name == 'xgboost':

#             self.y_pred_test = self.model.predict(
#                 xgb.DMatrix(self.split['X_test'][self.var])
#             )
#             y_pred_test_df = pd.DataFrame(
#                 data = self.y_pred_test,
#                 columns = ['snow_depth_pred']
#             )

#             self.y_pred_train = self.model.predict(
#                 xgb.DMatrix(self.split['X_temp'][self.var])
#             )
#             y_pred_train_df = pd.DataFrame(
#                 data = self.y_pred_train,
#                 columns = ['snow_depth_pred']
#             )

#             return {
#                 'y_pred_test': y_pred_test_df,
#                 'y_pred_train': y_pred_train_df
#             }

#         elif self.model_name == 'pytorch_nn':

#             self.predictions_test = predict(
#                 model=self.model,
#                 test_loader=self.loader['test_dataloader'],
#                 device=device
#             )

#             self.predictions_train = predict(
#                 model=self.model,
#                 test_loader=self.loader['train_dataloader'],
#                 device=device
#             )

#             y_pred_test_df = pd.DataFrame(
#                 data = self.predictions_test['predictions'],
#                 columns = ['snow_depth_pred']
#             )

#             y_pred_train_df = pd.DataFrame(
#                 data = self.predictions_train['predictions'],
#                 columns = ['snow_depth_pred']
#             )

#             return {
#                 'y_pred_test': y_pred_test_df,
#                 'y_pred_train': y_pred_train_df
#             }

#         elif self.model_name == 'linear_regression':
#             self.y_pred_test = self.model.predict(self.split['X_test'][self.var])
#             y_pred_test_df = pd.DataFrame(
#                 data = self.y_pred_test,
#                 columns = ['snow_depth_pred']
#             )

#             self.y_pred_train = self.model.predict(self.split['X_temp'][self.var])
#             y_pred_train_df = pd.DataFrame(
#                 data = self.y_pred_train,
#                 columns = ['snow_depth_pred']
#             )

#             return {
#                 'y_pred_test': y_pred_test_df,
#                 'y_pred_train': y_pred_train_df
#             }

#         elif self.model_name == 'random_forest': # New: Make Random Forest predictions
#             self.y_pred_test = self.model.predict(self.split['X_test'][self.var])
#             y_pred_test_df = pd.DataFrame(
#                 data = self.y_pred_test,
#                 columns = ['snow_depth_pred']
#             )

#             self.y_pred_train = self.model.predict(self.split['X_temp'][self.var])
#             y_pred_train_df = pd.DataFrame(
#                 data = self.y_pred_train,
#                 columns = ['snow_depth_pred']
#             )

#             return {
#                 'y_pred_test': y_pred_test_df,
#                 'y_pred_train': y_pred_train_df
#             }

#         else:
#             raise ValueError(f'Invalid model name: {self.model_name}. \nPlease choose from: extra_trees, xgboost, pytorch_nn, linear_regression, random_forest.')


#     def evaluate_model(self) -> pd.DataFrame:

#         """
#         A function that evaluates the performance of a model using the RMSE, MBE and R2 metrics.
#         """

#         if self.model_name == 'extra_trees':

#             train_eval=compute_metrics(
#                 y_true=self.split['y_temp'],
#                 y_pred=self.y_pred_train,
#                 model_name=self.model_name + '_train'
#             )

#             test_eval=compute_metrics(
#                 y_true=self.split['y_test'],
#                 y_pred=self.y_pred_test,
#                 model_name=self.model_name + '_test'
#             )

#             eval_df = pd.concat([train_eval, test_eval], axis=1)
#             display(eval_df)

#             return eval_df

#         elif self.model_name == 'xgboost':

#             train_eval=compute_metrics(
#                 y_true=self.split['y_temp'],
#                 y_pred=self.y_pred_train,
#                 model_name=self.model_name + '_train'
#             )

#             test_eval=compute_metrics(
#                 y_true=self.split['y_test'],
#                 y_pred=self.y_pred_test,
#                 model_name=self.model_name + '_test'
#             )

#             eval_df = pd.concat([train_eval, test_eval], axis=1)
#             display(eval_df)

#             return eval_df

#         elif self.model_name == 'pytorch_nn':

#             train_eval=compute_metrics(
#                 y_true=self.predictions_train['labels'],
#                 y_pred=self.predictions_train['predictions'],
#                 model_name=self.model_name + '_train'
#             )

#             test_eval=compute_metrics(
#                 y_true=self.predictions_test['labels'],
#                 y_pred=self.predictions_test['predictions'],
#                 model_name=self.model_name + '_test'
#             )

#             eval_df = pd.concat([train_eval, test_eval], axis=1)
#             display(eval_df)

#             return eval_df

#         elif self.model_name == 'linear_regression':
#             train_eval=compute_metrics(
#                 y_true=self.split['y_temp'],
#                 y_pred=self.y_pred_train,
#                 model_name=self.model_name + '_train'
#             )

#             test_eval=compute_metrics(
#                 y_true=self.split['y_test'],
#                 y_pred=self.y_pred_test,
#                 model_name=self.model_name + '_test'
#             )

#             eval_df = pd.concat([train_eval, test_eval], axis=1)
#             display(eval_df)

#             return eval_df

#         elif self.model_name == 'random_forest': # New: Evaluate Random Forest
#             train_eval=compute_metrics(
#                 y_true=self.split['y_temp'],
#                 y_pred=self.y_pred_train,
#                 model_name=self.model_name + '_train'
#             )

#             test_eval=compute_metrics(
#                 y_true=self.split['y_test'],
#                 y_pred=self.y_pred_test,
#                 model_name=self.model_name + '_test'
#             )

#             eval_df = pd.concat([train_eval, test_eval], axis=1)
#             display(eval_df)

#             return eval_df

#         else:
#             raise ValueError(f'Invalid model name: {self.model_name}. \nPlease choose from: extra_trees, xgboost, pytorch_nn, linear_regression, random_forest.')


#     def get_importance(self) -> pd.DataFrame:

#         if self.model_name == 'extra_trees':

#             feature_importance = pd.DataFrame(
#                 data = {
#                     'feature': self.split['X_temp'][self.var].columns,
#                     'importance': self.model.feature_importances_
#                 }
#             ).sort_values(by='importance', ascending=False)

#             display(feature_importance)

#             return feature_importance

#         elif self.model_name == 'xgboost':

#             xgb_importance = self.model.get_score(importance_type='gain')
#             feature_importance_list = []
#             for col in self.split['X_temp'][self.var].columns:
#                 feature_importance_list.append({'feature': col, 'importance': xgb_importance.get(col, 0)})

#             feature_importance = pd.DataFrame(feature_importance_list).sort_values(by='importance', ascending=False)

#             display(feature_importance)

#             return feature_importance

#         elif self.model_name == 'pytorch_nn':
#             print('No direct feature importance for PyTorch NNs in this setup.')
#             return pd.DataFrame()

#         elif self.model_name == 'linear_regression':
#             feature_importance = pd.DataFrame(
#                 data = {
#                     'feature': self.split['X_temp'][self.var].columns,
#                     'importance': self.model.coef_
#                 }
#             ).sort_values(by='importance', ascending=False)

#             display(feature_importance)

#             return feature_importance

#         elif self.model_name == 'random_forest': # New: Get feature importance for Random Forest
#             feature_importance = pd.DataFrame(
#                 data = {
#                     'feature': self.split['X_temp'][self.var].columns,
#                     'importance': self.model.feature_importances_
#                 }
#             ).sort_values(by='importance', ascending=False)

#             display(feature_importance)

#             return feature_importance

#         else:
#             raise ValueError(f'Invalid model name: {self.model_name}.')
