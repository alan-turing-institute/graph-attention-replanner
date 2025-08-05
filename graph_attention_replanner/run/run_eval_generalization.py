import subprocess
import sys
from tqdm.contrib.concurrent import process_map
from itertools import product

"""
This script streamline the evaluation process for the Generalisation Experiments. Usage is self-explanatory. If compute do not allow, please comment out sections and run step by step. Must generate data first (Step 1 below) before running the rest of the evaluation script.

Note that on Mac please change the run command code to:
for cmd in cmds:
    run_process(cmd)

Command:
    # This will print out the commands to run, once you are happy with the commands, uncomment the process_map line to run them.
    python run_eval.py
"""

N_JOB = 6  # Set to number of cpu cores

# Phase 2
exp_settings = {
    "--seed": [10, 11, 12],
    "--mtsp_problem_type": [5],
    "--num_agent": [2, 4, 6],
    "--num_task": [2, 4, 6],
    "--discretize_level": [1, 2, 4],
    "--batch_size": [10000],
}
pairwise_exp_settings = {}
data_seeds = [10, 11, 12]
# Which model to evaulate on which data, here model 0 evaulate on data 10
method_seeds = {
    10: 0,
    11: 1,
    12: 2,
}


def get_exp_settings():
    pairwise_combinations = get_pairwise_combinations(pairwise_exp_settings)
    normal_combinations = get_all_combinations(exp_settings)
    combinations = combine_exp_settings(pairwise_combinations, normal_combinations)
    for idx, combination in enumerate(combinations):
        print(f"\nExperiment {idx} Parameters Under Change: {combination}")
    settings = []
    for combination in combinations:
        cmd = []
        for key, value in combination.items():
            cmd.append(key)
            cmd.append(str(value))
        settings.append(cmd)
    return settings


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
        param_dict["--num_node"] = numtask2node(
            param_dict["--num_task"], param_dict["--discretize_level"]
        )  # TODO:Temp. To enforce no padded model
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


def run_process(cmd):
    process = subprocess.Popen(cmd)
    process.wait()


def print_cmd(cmds):
    num_cmd = len(cmds) - 1
    for i, cmd in enumerate(cmds):
        print(f"\n Commands {i}/ {num_cmd}: {cmd}")
        print(" ".join(cmd))


settings = get_exp_settings()

# Step 1
# Generate neccessary data
cmds = []
for data_seed in data_seeds:
    for setting_arg in settings:
        cmd = [sys.executable, "gen_data.py"] + setting_arg
        cmds += [cmd]
print_cmd(cmds)
# process_map(run_process, cmds, max_workers=N_JOB, desc="Generating Data", leave=True)

# Step 2
# Generate neccessary data for LKH
cmds = []
for setting_arg in settings:
    cmd = [
        sys.executable,
        "gen_data_td2tsp.py",
        "--method_mtsp_problem_type",
        "5",
        "--num_exp",
        "10000",
    ] + setting_arg
    cmds += [cmd]
print_cmd(cmds)
# process_map(run_process, cmds, max_workers=N_JOB, desc="Generating Data", leave=True)

# Step 3
# Evaluation
methods = [
    ("gat", "../method/gat/eval_gat_generalise.py"),
    ("gat_gen", "../method/gat/eval_gat.py"),
    (
        "random",
        "../method/gat/eval_random.py",
    ),
]
cmds = []
for data_seed in data_seeds:
    for setting_arg in settings:
        for method, script in methods:
            cmd = [
                sys.executable,
                script,
                "--method_seed",
                str(method_seeds[data_seed]),
                "--seed",
                str(data_seed),
                "--method_mtsp_problem_type",
                "5",
                "--num_exp",
                "10000",
            ] + setting_arg
            cmds += [cmd]

methods = [
    ("lkh", "../method/lkh/eval_lkh.py"),
]
for data_seed in data_seeds:
    for setting_arg in settings:
        for method, script in methods:
            cmd = [
                sys.executable,
                script,
                "--seed",
                str(data_seed),
                "--method_mtsp_problem_type",
                "5",
                "--num_exp",
                "10000",
            ] + setting_arg
            cmds += [cmd]
print_cmd(cmds)
# process_map(run_process, cmds, max_workers=N_JOB, desc="Evaluating", leave=True)
