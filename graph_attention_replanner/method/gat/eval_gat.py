import torch
import time
import numpy as np
import pandas as pd
from rl4co.models import REINFORCE

from graph_attention_replanner.config import (
    MissionPlanConfig,
    LogFileConfig,
    get_generator,
    get_env,
)

"""
This script evaluate GATR General models. 
Always put the spec of problem 5 as args, then based on the actual problem type we want to solve, we change mtsp_problem_type, this ensure we load the right data file.

For help:
python eval_gat.py --help

Usage:
python eval_gat.py 
# method param
--method_mtsp_problem_type 1 --method_seed 0
# data param
--mtsp_problem_type 5 --num_node 12 --num_task 4 --discretize_level 3 --num_agent 3 --exact_num_node --batch_size 1000 --num_exp 1

Example Command:
python eval_gat.py --method_mtsp_problem_type 1 --method_seed 0 --mtsp_problem_type 5 --num_node 12 --num_task 4 --discretize_level 3 --num_agent 3 --exact_num_node --batch_size 1000 --num_exp 1
"""


def get_model_cfg(args):
    # convert_dataspec_to_modelspec_based_on_problem_type
    if args.method_mtsp_problem_type <= 4:
        min_discretize_level = 1
        max_discretize_level = 1
        num_node = args.num_task
    else:
        min_discretize_level = args.min_discretize_level
        max_discretize_level = args.max_discretize_level
        num_node = args.num_node

    model_cfg = LogFileConfig(
        args.method_mtsp_problem_type,
        num_node,
        args.min_num_task,
        args.max_num_task,
        min_discretize_level,
        max_discretize_level,
        args.min_num_agent,
        args.max_num_agent,
        args.batch_size,
    )
    return model_cfg


def main():
    args = MissionPlanConfig().parse_eval_args()
    args.num_node_incl_home = args.num_node + 1

    model_cfg = get_model_cfg(args)
    model_path = model_cfg.get_train_modelpath(seed=args.method_seed)

    data_cfg = LogFileConfig(
        args.mtsp_problem_type,
        args.num_node,
        args.min_num_task,
        args.max_num_task,
        args.min_discretize_level,
        args.max_discretize_level,
        args.min_num_agent,
        args.max_num_agent,
        args.batch_size,
    )
    try:
        result_path = data_cfg.get_result_logfilename(
            method="gat",
            format="csv",
            override_mtsp_problem_type=args.method_mtsp_problem_type,
            seed=args.seed,
            method_seed=args.method_seed,
        )
    except FileExistsError:
        return
    data_path = data_cfg.get_data_logfilename(
        seed=args.seed, format="npz", override_mtsp_problem_type=args.mtsp_problem_type
    )  # Use Problem 5 (largest) data, and trim it down to smaller problem data))
    print(
        f"Evaluating model from {model_path} using data from {data_path} and saving results at {result_path} ..."
    )

    # Set up device
    device = torch.device(
        f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
    )

    model = REINFORCE.load_from_checkpoint(
        model_path, load_baseline=False, strict=False
    )
    model = model.to(device)

    # Load dataset. set generator and env based on the problem type of the dataset
    data_generator = get_generator(args.mtsp_problem_type)(
        num_node=args.num_node,
        min_num_task=args.min_num_task,
        max_num_task=args.max_num_task,
        min_discretize_level=args.min_discretize_level,
        max_discretize_level=args.max_discretize_level,
        min_num_agent=args.min_num_agent,
        max_num_agent=args.max_num_agent,
        seed=args.seed,
    )
    data_env = get_env(args.mtsp_problem_type)(data_generator)
    data_td = data_env.load_data(data_path, batch_size=args.batch_size).to(device)

    # We assume the dataset is from problem 5.
    # Locations
    if args.method_mtsp_problem_type == 1:
        data_td["locs"] = data_td["locs"][:, : (1 + args.num_task)]
    elif args.method_mtsp_problem_type == 2:
        data_td["locs"] = data_td["locs"][:, : (1 + args.num_task)]
        # generator 4 doesnt have this key
        data_td["unsplited_task_length"] = (
            data_td["unsplited_task_length"]
            if "unsplited_task_length" in data_td
            else data_td["task_length"]
        )

        data_td["task_length"] = data_td["unsplited_task_length"][
            :, : (1 + args.num_task)
        ]

    elif args.method_mtsp_problem_type == 3:
        # taking_unsplitedloc_as_loc
        start_locs = data_td["locs"][
            :, (1 + args.num_node) : (1 + args.num_node + args.max_num_agent)
        ]  # start_loc
        data_td["locs"] = data_td["locs"][
            :, : (1 + args.num_task + args.max_num_agent)
        ]  # Setting the size of the table
        data_td["locs"][:, : (1 + args.num_task)] = data_td["locs"][
            :, : (1 + args.num_task)
        ]  # loc
        data_td["locs"][:, (1 + args.num_task) :] = start_locs  # start_loc
    elif args.method_mtsp_problem_type == 4 and args.mtsp_problem_type == 5:
        # taking_unsplitedloc_as_loc
        start_locs = data_td["locs"][
            :, (1 + args.num_node) : (1 + args.num_node + args.max_num_agent)
        ]  # start_loc
        data_td["locs"] = data_td["locs"][
            :, : (1 + args.num_task + args.max_num_agent)
        ]  # Setting the size of the table
        data_td["locs"][:, : (1 + args.num_task)] = data_td["locs"][
            :, : (1 + args.num_task)
        ]  # loc
        data_td["locs"][:, (1 + args.num_task) :] = start_locs  # start_loc

        # generator 4 doesnt have this key
        data_td["unsplited_task_length"] = (
            data_td["unsplited_task_length"]
            if "unsplited_task_length" in data_td
            else data_td["task_length"]
        )
        data_td["task_length"] = data_td["task_length"][
            :, : (1 + args.num_task + args.max_num_agent)
        ]  # Setting the size
        data_td["task_length"][:, : (1 + args.num_task)] = data_td[
            "unsplited_task_length"
        ][:, : (1 + args.num_task)]  # Copying Task Length
        data_td["task_length"][:, (1 + args.num_task) :] = torch.zeros(
            args.max_num_agent
        )  # Start loc task length are zeros

    if args.method_mtsp_problem_type != args.mtsp_problem_type:
        # Set generator and env based on the problem type we want to solve for.
        if args.method_mtsp_problem_type <= 4:
            method_num_node = args.num_task
            method_discretize_level = 1
        else:
            method_num_node = args.num_node
            method_discretize_level = args.discretize_level
        method_generator = get_generator(args.method_mtsp_problem_type)(
            num_node=method_num_node,
            min_num_task=args.min_num_task,
            max_num_task=args.max_num_task,
            min_discretize_level=method_discretize_level,
            max_discretize_level=method_discretize_level,
            min_num_agent=args.min_num_agent,
            max_num_agent=args.max_num_agent,
            seed=args.seed,
        )
        method_env = get_env(args.method_mtsp_problem_type)(method_generator)
        method_td = method_env.generator(args.batch_size).to(device)
        for key in method_td.keys():
            method_td[key] = data_td[key]
        td_init = method_env.reset(method_td)

    else:
        td_init = data_env.reset(data_td)

    print(f"------------Loaded Dataset from {data_path}-------------")

    # Evaluate the data set generated in the td_init variable
    start_time = time.time()
    out = model(td_init, phase="test", decode_type="greedy", return_actions=True)
    end_time = time.time()
    execution_time = end_time - start_time

    print("\n\n\n------------------------------------------")
    mission_times = -out["reward"]
    sd, mean = torch.std_mean(mission_times)
    se = sd / np.sqrt(mission_times.size(dim=0))
    print(f"Avg Mission Time(s): {mean} ± {sd}. \nStandard Error: {se}")

    avg_runtime = execution_time / args.batch_size
    print(f"Avg Wall Time(s): {avg_runtime}")
    print("------------------------------------------\n\n\n")

    # Save results to csv
    mt = mission_times.cpu().detach().numpy()
    num_data = mt.shape[0]
    a = out["actions"].cpu().detach().tolist()  # must be to list to save it properly
    results_dict = {
        "data_path": np.repeat(data_path, num_data),
        "mtsp_problem_type": np.repeat(args.method_mtsp_problem_type, num_data),
        "exp_idx": np.arange(num_data),
        "mission_time": mt,
        "tour": a,
        "runtime": np.repeat(avg_runtime, num_data),
        "model_seed": np.repeat(args.method_seed, num_data),
    }
    df = pd.DataFrame(results_dict)
    df.to_csv(result_path)
    return

if __name__ == "__main__":
    main()
