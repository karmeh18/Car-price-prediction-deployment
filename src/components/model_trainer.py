import os
import sys
import numpy as np
from dataclasses import dataclass
import optuna
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor

from src.logger import logging
from src.exception import Custom_Exception
from src.utils import calculate_mape, save_object, evaluate_models


@dataclass
class ModelTrainer:
    def initiate_model_trainer(self,train_arr,test_arr):
        # Get absolute path for cross-platform compatibility
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_trainer_config = os.path.join(project_root, 'artifacts', "model.pkl")

        try:
            logging.info("Splitting Data into Training Data and Test Data")
            X_train,y_train,X_test,y_test=(train_arr[:,:-1],
                                           train_arr[:,-1],
                                           test_arr[:,:-1],
                                           test_arr[:,-1])
            models={"LinearRegression":LinearRegression(),
                    "Ridge Regression":Ridge(),
                    "Lasso Regression":Lasso(),
                    "DecisionTreeRegressor":DecisionTreeRegressor(),
                    "RandomForestRegressor":RandomForestRegressor(),
                    "Gradient Boosting Regressor":GradientBoostingRegressor(),
                    "Support Vector Regression":SVR(),
                    "XGB Regressor":XGBRegressor(),
                    "Cat Boost Regressor":CatBoostRegressor(),
                    "Light GBM":LGBMRegressor(),
                    "MLP Regressor":MLPRegressor()
                    }
            model_report=evaluate_models(X_train,y_train,X_test,y_test,models)
            logging.info("Model Training has been done and now selecting the best model")
            best_model_score=min(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise Custom_Exception("No best Model Found")
            logging.info("Best Model has been found and the model is '{}'".format(best_model_name))

            predicted=best_model.predict(X_test)
            score=calculate_mape(y_test,predicted)
            r2=r2_score(y_test,predicted)
            logging.info("Since the best model has been  discovered now combining the training data and test data into one complete dataset before pushing for the Model Deployment")
            logging.info("Model with r2 score is {} and MAPE score is {}".format(r2,score))
            arr1=np.vstack((X_train,X_test))
            arr2=np.vstack((y_train.reshape(-1,1),y_test.reshape(-1,1)))
            logging.info("Concatention of array datapoints from training data and test data has been completed")
            best_model.fit(arr1,arr2)
            logging.info("Best model has been fit on the complete data")
            save_object(file_path=self.model_trainer_config,obj=best_model)
            logging.info("Best model has been saved in artifact folder with path as '{}'".format(self.model_trainer_config))
            print(best_model_name)

            # Ask user if they want to proceed with hyperparameter optimization
            optimize_hyperparameters = input("Do you want to proceed with hyperparameter optimization? (yes/no): ")
            if optimize_hyperparameters.lower() == 'yes' or optimize_hyperparameters.lower()=='y':
                # Call a function to perform hyperparameter optimization
                logging.info("Optuna is running for Hyperparameter Optimization...")
                best_params=self.perform_hyperparameter_optimization(X_train, y_train,X_test,y_test)
                logging.info("Optuna has finished running Hyperparameter optimization")
                print(f"Best hyperparameters found: {best_params}")
                logging.info("Here are the best Hyper Parameter Optimization for {} ----- {}".format(best_model_name,best_params))
                
                # Ask user if they want to retrain with best parameters on full dataset
                retrain_with_full = input("\nRetrain model on FULL DATASET with best parameters? (yes/no): ")
                if retrain_with_full.lower() == 'yes' or retrain_with_full.lower() == 'y':
                    logging.info("Retraining model on full dataset with optimized parameters...")
                    print("\nRetraining on full dataset with optimized parameters...")
                    final_score = self.retrain_with_optimized_params(X_train, y_train, X_test, y_test, best_params, best_model_name)
                    self.save_best_params(best_params)
                    logging.info(f"Retraining completed. Final MAPE on test set: {final_score:.4f}")
                    print(f"✓ Retraining completed. Final MAPE: {final_score:.4f}")
                    score = final_score

            logging.info("Execution has been completed")    
            return score

        except Exception as e:
            raise Custom_Exception(e, sys)
    def perform_hyperparameter_optimization(self, X_train, y_train, X_test, y_test):
        logging.info("Starting Hyperparameter Optimization with Optuna...")
        
        # Track best value and consecutive trials without improvement
        best_value_seen = [float('inf')]
        trials_without_improvement = [0]
        patience_trials = 5  # Need 5+ trials without improvement to stop
        min_trials = 8  # Always run at least 8 trials
        
        def early_stopping_callback(study, trial):
            """
            Improved early stopping:
            - If best comes at trial 4, continue for 5-6 more trials
            - Only stop if 5+ consecutive trials show no improvement
            - Minimum 8 trials always
            """
            current_best = study.best_value
            n_trials = len(study.trials)
            
            # Update best value tracking
            if current_best < best_value_seen[0]:
                best_value_seen[0] = current_best
                trials_without_improvement[0] = 0
                logging.info(f"Trial {n_trials}: New best value: {current_best:.4f}")
            else:
                trials_without_improvement[0] += 1
            
            # Stop conditions:
            # 1. Minimum 8 trials must complete
            # 2. 5+ trials without improvement
            if n_trials >= min_trials and trials_without_improvement[0] >= patience_trials:
                study.stop()
                logging.warning(
                    f"Early stopping at trial {n_trials}: "
                    f"{patience_trials} consecutive trials without improvement. "
                    f"Best MAPE: {study.best_value:.4f}"
                )
                print(
                    f"Early stopping at trial {n_trials}: "
                    f"{patience_trials} trials without improvement. "
                    f"Best value: {study.best_value:.4f}"
                )
        
        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_test, y_test),
            n_trials=50,
            callbacks=[early_stopping_callback]
        )
        
        best_params = study.best_params
        best_value = study.best_value
        n_trials = len(study.trials)
        
        logging.info(f"Hyperparameter Optimization completed after {n_trials} trials")
        logging.info(f"Best MAPE score achieved: {best_value:.4f}")
        logging.info(f"Best parameters found:")
        logging.info(f"  - n_estimators: {best_params.get('n_estimators')}")
        logging.info(f"  - max_depth: {best_params.get('max_depth')}")
        logging.info(f"  - criterion: {best_params.get('criterion')}")
        
        print(f"\nOptuna Optimization Results:")
        print(f"Total trials run: {n_trials}")
        print(f"Best MAPE: {best_value:.4f}")
        print(f"Best Parameters: {best_params}\n")
        
        return best_params


    def objective(self,trial,X_train,y_train,X_test,y_test):
        try:
            # Add code here to perform hyperparameter optimization
            # Define the hyperparameters to optimize
            n_estimators = trial.suggest_int('n_estimators', 10, 200)
            max_depth = trial.suggest_int('max_depth', 1, 20)
            criterion = trial.suggest_categorical('criterion', ['squared_error','absolute_error','friedman_mse','poisson'])
            #min_samples_leaf = trial.suggest_float('subsample', 0.2, 1.0)
            # Initialize and train the model with the suggested hyperparameters
            model = RandomForestRegressor(
                n_estimators=n_estimators,  # Number of iterations (equivalent to n_estimators in other algorithms)
                max_depth=max_depth,  # Maximum depth of trees
                criterion=criterion,  # criterion
                random_state=42,  # Random seed for reproducibility
                #verbose=0  # To disable RandomForest Regressor log messages
            )
            model.fit(X_train,y_train)
            
            #Make Prediction on test set
            y_pred=model.predict(X_test)
            
            #Calculate MAPE and r2 score 
            mae_optuna=mean_absolute_error(y_test,y_pred)
            r2_optuna=r2_score(y_test,y_pred)
            mape_optuna=calculate_mape(y_test,y_pred)

            return mape_optuna
            print(mae_optuna,r2_optuna)
            logging.info("Hyperparameter optimization in progress...")
        except Exception as e:
            raise Custom_Exception(e, sys)
    
    def retrain_with_optimized_params(self, X_train, y_train, X_test, y_test, best_params, model_name):
        """
        Retrain model on full dataset using best hyperparameters
        Part of the pipeline after hyperparameter optimization
        """
        try:
            from src.utils import calculate_mape
            
            # Combine train and test (full dataset)
            X_full = np.vstack((X_train, X_test))
            y_full = np.vstack((y_train.reshape(-1, 1), y_test.reshape(-1, 1))).ravel()
            
            logging.info(f"Combining datasets for full model training")
            logging.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}, Full dataset: {len(X_full)}")
            
            # Create model with best parameters
            optimized_model = RandomForestRegressor(
                n_estimators=best_params.get('n_estimators', 100),
                max_depth=best_params.get('max_depth', 10),
                criterion=best_params.get('criterion', 'squared_error'),
                random_state=42
            )
            
            logging.info(f"Training {model_name} on full dataset with optimized parameters")
            logging.info(f"  - n_estimators: {best_params.get('n_estimators')}")
            logging.info(f"  - max_depth: {best_params.get('max_depth')}")
            logging.info(f"  - criterion: {best_params.get('criterion')}")
            
            print(f"\nTraining on full dataset ({len(X_full)} samples)...")
            print(f"  - n_estimators: {best_params.get('n_estimators')}")
            print(f"  - max_depth: {best_params.get('max_depth')}")
            print(f"  - criterion: {best_params.get('criterion')}")
            
            # Train on full dataset
            optimized_model.fit(X_full, y_full)
            
            # Evaluate on test set
            y_pred = optimized_model.predict(X_test)
            mape_score = calculate_mape(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logging.info(f"Full dataset training completed - MAPE: {mape_score:.4f}, R²: {r2:.4f}")
            print(f"\nFull Dataset Training Results:")
            print(f"  - MAPE: {mape_score:.4f}")
            print(f"  - R² Score: {r2:.4f}")
            
            # Save retrained model
            save_object(file_path=self.model_trainer_config, obj=optimized_model)
            logging.info("Optimized model trained on full dataset and saved")
            
            return mape_score
            
        except Exception as e:
            raise Custom_Exception(e, sys)
    
    def save_best_params(self, best_params):
        """Save best parameters to JSON file for reference"""
        import json
        try:
            params_file = os.path.join('artifacts', 'best_hyperparams.json')
            with open(params_file, 'w') as f:
                json.dump(best_params, f, indent=4)
            logging.info(f"Best hyperparameters saved to: {params_file}")
            print(f"✓ Best parameters saved to: {params_file}")
        except Exception as e:
            raise Custom_Exception(e, sys)
