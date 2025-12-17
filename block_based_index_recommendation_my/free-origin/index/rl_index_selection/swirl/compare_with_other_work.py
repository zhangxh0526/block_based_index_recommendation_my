import sys
sys.path.append('..')
import copy
import importlib
import logging
import pickle
import sys
#import gym_db  # noqa: F401
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize, sync_envs_normalization
from sb3_contrib import MaskablePPO
from gym_db.common import EnvironmentType
from custom_callback import EvalCallbackWithTBRunningAverage
#from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from compare_experiment import Experiment
from block_based_feature_extractor import CustomCombinedExtractor

# Compare RandomAll
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    CONFIGURATION_FILE = "../experiments/tpch.json"

    print("****************************Random Workload***************************************")

    experiment = Experiment(CONFIGURATION_FILE)

    # set parameter:
    experiment.config["workload"]["validation_testing"]["number_of_workloads"] = 3
    experiment.config["workload"]["test_number_of_workloads"] = 0
    experiment.config["workload"]["size"] = 16
    experiment.config["workload"]["excluded_query_classes"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22]
    experiment.config["workload"]["training_instances"] = 0
    violation_queries = set([])

    experiment_folder_path = "../experiment_results/ID_TPCH_Test_Experiment_timetamps_1694586398"

    experiment.prepare(experiment_folder_path, violation_queries)

    experiment.model_type = MaskablePPO

    with open(f"{experiment.experiment_folder_path}/experiment_object.pickle", "wb") as handle:
        pickle.dump(experiment, handle, protocol=pickle.HIGHEST_PROTOCOL)

    experiment.set_model()

    experiment.compare()

    experiment.finish()

    print("****************************Selectivity 0.00001***************************************")

    experiment = Experiment(CONFIGURATION_FILE)

    # set parameter:
    experiment.config["workload"]["validation_testing"]["number_of_workloads"] = 3
    experiment.config["workload"]["test_number_of_workloads"] = 0
    experiment.config["workload"]["size"] = 16
    experiment.config["workload"]["excluded_query_classes"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                                               22]
    experiment.config["workload"]["training_instances"] = 0
    violation_queries = set([])

    experiment_folder_path = "../experiment_results/ID_TPCH_Test_Experiment_timetamps_1694586398"

    experiment.prepare(experiment_folder_path, violation_queries)

    experiment.model_type = MaskablePPO

    with open(f"{experiment.experiment_folder_path}/experiment_object.pickle", "wb") as handle:
        pickle.dump(experiment, handle, protocol=pickle.HIGHEST_PROTOCOL)

    experiment.set_model()

    experiment.compare()

    experiment.finish()

    print("****************************Selectivity 0.0001***************************************")

    experiment = Experiment(CONFIGURATION_FILE)

    # set parameter:
    experiment.config["workload"]["validation_testing"]["number_of_workloads"] = 3
    experiment.config["workload"]["test_number_of_workloads"] = 0
    experiment.config["workload"]["size"] = 16
    experiment.config["workload"]["excluded_query_classes"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
                                                               18, 19, 20, 21,
                                                               22]
    experiment.config["workload"]["training_instances"] = 0
    violation_queries = set([])

    experiment_folder_path = "../experiment_results/ID_TPCH_Test_Experiment_timetamps_1694586398"

    experiment.prepare(experiment_folder_path, violation_queries)

    experiment.model_type = MaskablePPO

    with open(f"{experiment.experiment_folder_path}/experiment_object.pickle", "wb") as handle:
        pickle.dump(experiment, handle, protocol=pickle.HIGHEST_PROTOCOL)

    experiment.set_model()

    experiment.compare()

    experiment.finish()

    print("****************************Selectivity 0.001***************************************")

    experiment = Experiment(CONFIGURATION_FILE)

    # set parameter:
    experiment.config["workload"]["validation_testing"]["number_of_workloads"] = 3
    experiment.config["workload"]["test_number_of_workloads"] = 0
    experiment.config["workload"]["size"] = 16
    experiment.config["workload"]["excluded_query_classes"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17,
                                                               18, 19, 20, 21,
                                                               22]
    experiment.config["workload"]["training_instances"] = 0
    violation_queries = set([])

    experiment_folder_path = "../experiment_results/ID_TPCH_Test_Experiment_timetamps_1694586398"

    experiment.prepare(experiment_folder_path, violation_queries)

    experiment.model_type = MaskablePPO

    with open(f"{experiment.experiment_folder_path}/experiment_object.pickle", "wb") as handle:
        pickle.dump(experiment, handle, protocol=pickle.HIGHEST_PROTOCOL)

    experiment.set_model()

    experiment.compare()

    experiment.finish()

    print("****************************Selectivity 0.01***************************************")

    experiment = Experiment(CONFIGURATION_FILE)

    # set parameter:
    experiment.config["workload"]["validation_testing"]["number_of_workloads"] = 3
    experiment.config["workload"]["test_number_of_workloads"] = 0
    experiment.config["workload"]["size"] = 16
    experiment.config["workload"]["excluded_query_classes"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17,
                                                               18, 19, 20, 21,
                                                               22]
    experiment.config["workload"]["training_instances"] = 0
    violation_queries = set([])

    experiment_folder_path = "../experiment_results/ID_TPCH_Test_Experiment_timetamps_1694586398"

    experiment.prepare(experiment_folder_path, violation_queries)

    experiment.model_type = MaskablePPO

    with open(f"{experiment.experiment_folder_path}/experiment_object.pickle", "wb") as handle:
        pickle.dump(experiment, handle, protocol=pickle.HIGHEST_PROTOCOL)

    experiment.set_model()

    experiment.compare()

    experiment.finish()

    print("****************************Selectivity 0.05***************************************")

    experiment = Experiment(CONFIGURATION_FILE)

    # set parameter:
    experiment.config["workload"]["validation_testing"]["number_of_workloads"] = 3
    experiment.config["workload"]["test_number_of_workloads"] = 0
    experiment.config["workload"]["size"] = 16
    experiment.config["workload"]["excluded_query_classes"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17,
                                                               18, 19, 20, 21,
                                                               22]
    experiment.config["workload"]["training_instances"] = 0
    violation_queries = set([])

    experiment_folder_path = "../experiment_results/ID_TPCH_Test_Experiment_timetamps_1694586398"

    experiment.prepare(experiment_folder_path, violation_queries)

    experiment.model_type = MaskablePPO

    with open(f"{experiment.experiment_folder_path}/experiment_object.pickle", "wb") as handle:
        pickle.dump(experiment, handle, protocol=pickle.HIGHEST_PROTOCOL)

    experiment.set_model()

    experiment.compare()

    experiment.finish()