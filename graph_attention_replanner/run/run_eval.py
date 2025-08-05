import subprocess
import sys
from tqdm.contrib.concurrent import process_map

"""
This script streamline the evaluation process for the Main Experiments (Building Up the Problem). Usage is self-explanatory. If compute do not allow, please comment out sections and run step by step. Must generate data first (Step 1 below) before running the rest of the evaluation script.

Note that on Mac please change the run command code to:
for cmd in cmds:
    run_process(cmd)

Command:
# This will print out the commands to run, once you are happy with the commands, uncomment the process_map line to run them.
python run_eval.py
"""

N_JOB = 6

dataset_4task_arg = [
    "--mtsp_problem_type",
    "5",
    "--num_node",
    "8",
    "--num_task",
    "4",
    "--discretize_level",
    "2",
    "--num_agent",
    "3",
    "--batch_size",
    "100",
    "--num_exp",
    "100",
]

dataset_3task_arg = [
    "--mtsp_problem_type",
    "5",
    "--num_node",
    "6",
    "--num_task",
    "3",
    "--discretize_level",
    "2",
    "--num_agent",
    "3",
    "--batch_size",
    "100",
    "--num_exp",
    "100",
]

dataset_2task_arg = [
    "--mtsp_problem_type",
    "5",
    "--num_node",
    "4",
    "--num_task",
    "2",
    "--discretize_level",
    "2",
    "--num_agent",
    "3",
    "--batch_size",
    "100",
    "--num_exp",
    "100",
]

dataset_6task_arg = [
    "--mtsp_problem_type",
    "4",
    "--num_node",
    "6",
    "--num_task",
    "6",
    "--discretize_level",
    "1",
    "--num_agent",
    "3",
    "--batch_size",
    "100",
    "--num_exp",
    "100",
]

dataset_8task_arg = [
    "--mtsp_problem_type",
    "4",
    "--num_node",
    "8",
    "--num_task",
    "8",
    "--discretize_level",
    "1",
    "--num_agent",
    "3",
    "--batch_size",
    "100",
    "--num_exp",
    "100",
]

dataset_env6_arg = [
    "--mtsp_problem_type",
    "6",
    "--num_node",
    "8",
    "--num_task",
    "8",
    "--discretize_level",
    "1",
    "--num_agent",
    "3",
    "--batch_size",
    "100",
    "--num_exp",
    "100",
]

dataset_env7_arg = [
    "--mtsp_problem_type",
    "7",
    "--num_node",
    "8",
    "--num_task",
    "8",
    "--discretize_level",
    "1",
    "--num_agent",
    "3",
    "--batch_size",
    "100",
    "--num_exp",
    "100",
]


datasets_arg = {
    "dataset_4task": dataset_4task_arg,
    "dataset_3task": dataset_3task_arg,
    "dataset_2task": dataset_2task_arg,
    "dataset_6task": dataset_6task_arg,
    "dataset_8task": dataset_8task_arg,
}

# method_mtsp_problem_type = {
#     "dataset_4task": [1, 2, 3, 4, 5],
#     "dataset_3task": [5],
#     "dataset_2task": [5],
#     "dataset_6task": [1, 2, 3, 4],
#     "dataset_8task": [1, 2, 3, 4],
# }

method_mtsp_problem_type = {
    "dataset_4task": [3, 4, 5],
    "dataset_3task": [5],
    "dataset_2task": [5],
    "dataset_6task": [3, 4],
    "dataset_8task": [3, 4],
}

data_seeds = [10, 11, 12]
# Which model to evaulate on which data, here model 0 evaulate on data 10
method_seeds = {
    10: 0,
    11: 1,
    12: 2,
}


def run_process(cmd):
    process = subprocess.Popen(cmd)
    process.wait()


def print_cmd(cmds):
    num_cmd = len(cmds) - 1
    for i, cmd in enumerate(cmds):
        print(f"\n Commands {i}/ {num_cmd}: {cmd}")
        print(" ".join(cmd))


# Step 1 Generate neccessary data
cmds = []
for data_seed in data_seeds:
    for dataset, dataset_arg in datasets_arg.items():
        cmd = [sys.executable, "gen_data.py", "--seed", str(data_seed)] + dataset_arg
        cmds += [cmd]
print_cmd(cmds)
# process_map(run_process, cmds, max_workers=N_JOB, desc="Generating Data", leave=True)

# Step 2 Generate neccessary data
cmds = []
for data_seed in data_seeds:
    for dataset, dataset_arg in datasets_arg.items():
        for mtsp_problem_type in method_mtsp_problem_type[dataset]:
            cmd = [
                sys.executable,
                "gen_data_td2tsp.py",
                "--seed",
                str(data_seed),
                "--method_mtsp_problem_type",
                str(mtsp_problem_type),
            ] + dataset_arg
            cmds += [cmd]
print_cmd(cmds)
# process_map(run_process, cmds, max_workers=N_JOB, desc="Generating Data", leave=True)


# Step 3
# Evaluate GATs and Random which both have the method_seed arguments
methods = [
    ("gat", "../method/gat/eval_gat.py"),
    ("random", "../method/gat/eval_random.py"),
]
cmds = []
for data_seed in data_seeds:
    for dataset, dataset_arg in datasets_arg.items():
        for mtsp_problem_type in method_mtsp_problem_type[dataset]:
            for method, script in methods:
                cmd = [
                    sys.executable,
                    script,
                    "--method_seed",
                    str(method_seeds[data_seed]),
                    "--seed",
                    str(data_seed),
                    "--method_mtsp_problem_type",
                    str(mtsp_problem_type),
                ] + dataset_arg
                cmds += [cmd]
print_cmd(cmds)
# process_map(run_process, cmds, max_workers=N_JOB, desc="Evaluating", leave=True)

# Evaluate the methods
methods = [
    ("lkh", "../method/lkh/eval_lkh.py"),
    ("enumeration", "../method/enumeration/eval_enum.py"),
]
cmds = []
for data_seed in data_seeds:
    for dataset, dataset_arg in datasets_arg.items():
        for mtsp_problem_type in method_mtsp_problem_type[dataset]:
            for method, script in methods:
                cmd = [
                    sys.executable,
                    script,
                    "--seed",
                    str(data_seed),
                    "--method_mtsp_problem_type",
                    str(mtsp_problem_type),
                ] + dataset_arg
                cmds += [cmd]
print_cmd(cmds)
# process_map(run_process, cmds, max_workers=N_JOB, desc="Evaluating", leave=True)

# Step 4
# Compiled median from enumeration results
cmds = []
for data_seed in data_seeds:
    for dataset, dataset_arg in datasets_arg.items():
        for mtsp_problem_type in method_mtsp_problem_type[dataset]:
            cmd = [
                sys.executable,
                "../method/enumeration/enum2median.py",
                "--seed",
                str(data_seed),
                "--method_mtsp_problem_type",
                str(mtsp_problem_type),
            ] + dataset_arg
            cmds += [cmd]
print_cmd(cmds)
# process_map(run_process, cmds, max_workers=N_JOB, desc="Evaluating", leave=True)
