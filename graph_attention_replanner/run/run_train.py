import subprocess
import sys
from tqdm.contrib.concurrent import process_map  # or thread_map
from itertools import product
import copy

"""
This script streamline the evaluation process for training the models needed in the Main Experiments (Building Up the Problem).

Command:
# This will print out the model to be trained, once you are happy with the commands, uncomment the process_map line to run them.
python run_train.py
"""

PYTHON_SCRIPT = "../method/gat/train.py"

# Main Experiments - Non Collab Cases
exp_settings = {
    "--seed": [1,2],
    "--mtsp_problem_type": [1, 2, 3, 4],
    "--discretize_level": [1],
    "--num_agent": [3],
}
pairwise_exp_settings = {
    "--num_node": [4, 6, 8],
    "--num_task": [4, 6, 8],
}
ENFORCE_NO_PADDING = False # Here we enforce it manually by specifying the num node

# Main Experiments - Collab Cases
# exp_settings = {
#     "--seed": [1,2],
#     "--mtsp_problem_type": [5],
#     "--discretize_level": [2],
#     "--num_agent": [3],
# }
# pairwise_exp_settings = {
#     "--num_node": [4, 6, 8],
#     "--num_task": [2, 3, 4],
# }
# ENFORCE_NO_PADDING = False # Here we enforce it manually by specifying the num node

BASE_CMD = [
    sys.executable,  # This will be the path to the current Python interpreter
    PYTHON_SCRIPT,
    "--batch_size",
    "5000",
    # "--batch_size",
    # "10",  # quick run for debug
    # "--train_data_size",
    # "100",  # quick run for debug
    # "--val_data_size",
    # "100",  # quick run for debug
    # "--max_epochs",
    # "2",  # quick run for debug
    # "--disable_wandb",  # quick run for debug
]


def get_pairwise_combinations(exp_settings):
    """
    Expand a dictionary of parameter lists into all possible pairwise combinations for grid search.
    """
    if not exp_settings:
        return exp_settings
    keys = list(exp_settings.keys())
    num_item = len(exp_settings[keys[0]])

    expanded_parameters = []
    for i in range(num_item):
        param_dict = {key: exp_settings[key][i] for key in keys}
        expanded_parameters.append(param_dict)
    # print(f"\n Parameters: {expanded_parameters}")
    return expanded_parameters


def get_all_combinations(exp_settings):
    """
    Expand a dictionary of parameter lists into all possible combinations for grid search.
    """
    keys = exp_settings.keys()
    value_lists = exp_settings.values()
    all_combinations = product(*value_lists)
    expanded_parameters = []
    for combination in all_combinations:
        param_dict = {key: value for key, value in zip(keys, combination)}
        if ENFORCE_NO_PADDING:
            param_dict["--num_node"] = numtask2node(
                param_dict["--num_task"], param_dict["--discretize_level"]
            )  # TODO:Temp. To enforce no padded model. Essentail for Phase 2 - Batch 1/ Phase 2 - Batch 2
        expanded_parameters.append(param_dict)
    return expanded_parameters


def numtask2node(num_task, discretize_level):
    return num_task * discretize_level


def combine_exp_settings(exp1, exp2):
    if not exp1:
        return exp2
    if not exp2:
        return exp1
    combinations = product(exp1, exp2)
    expanded_parameters = [
        {**combination[0], **combination[1]} for combination in combinations
    ]
    return expanded_parameters


def gen_cmd(extra_param):
    cmd = copy.deepcopy(BASE_CMD)
    for key, value in extra_param.items():
        cmd.append(key)
        cmd.append(str(value))
    return cmd


def run_process(cmd):
    process = subprocess.Popen(cmd)
    process.wait()


def main():
    pairwise_combinations = get_pairwise_combinations(pairwise_exp_settings)
    normal_combinations = get_all_combinations(exp_settings)
    combinations = combine_exp_settings(pairwise_combinations, normal_combinations)

    for idx, combination in enumerate(combinations):
        print(f"\nExperiment {idx} Parameters Under Change: {combination}")

    cmds = []
    for combination in combinations:
        cmd = gen_cmd(combination)
        cmds += [cmd]

    n_job = 6  # Set to number of cpu cores
    # process_map(run_process, cmds, max_workers=n_job, desc="Training", leave=True)


if __name__ == "__main__":
    main()
