import torch
import time
import numpy as np
import pandas as pd
from rl4co.models import REINFORCE
from tqdm.auto import tqdm

from graph_attention_replanner.config import (
    MissionPlanConfig,
    LogFileConfig,
    get_generator,
    get_env,
)


"""
This script collects more accurate runtime of GATR models (Collect runtime of every single data instance). Always put the spec of problem 5 as args, then based on the actual problem type we want to solve, we change mtsp_problem_type, this ensure we load the right data file.

For help:
python eval_gat_runtime.py --help

Usage:
python eval_gat_runtime.py
# method param
--method_mtsp_problem_type 1 --method_seed 0
# data param
--mtsp_problem_type 5 --num_node 12 --num_task 4 --discretize_level 3 --num_agent 3 --exact_num_node --batch_size 1000 --num_exp 1

Example Command:
python eval_gat_runtime.py --method_mtsp_problem_type 1 --method_seed 0 --mtsp_problem_type 5 --num_node 12 --num_task 4 --discretize_level 3 --num_agent 3 --exact_num_node --batch_size 1000 --num_exp 1
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

    assert (
        args.method_mtsp_problem_type == 5 and args.mtsp_problem_type == 5
    ), "We assume the model and dataset is from problem 5. Not supported for other problems."
    td_init = data_env.reset(data_td)

    # Evaluate the data set generated in the td_init variable
    runtimes = []
    mission_times = []
    actions = []
    small_env = get_env(args.mtsp_problem_type)(data_generator)

    for bs in tqdm(
        range(args.batch_size), desc=f"Running {args.batch_size} samples one by one"
    ):
        # setup small td_init
        small_td = data_env.generator(1).to(device)
        for key in small_td.keys():
            # print(key)
            # print(td_init[key].shape)
            # print(small_td[key].shape)
            if small_td[key][0].shape != td_init[key][bs].shape:
                # print(f"Mismatch in shape for {key}")
                small_td[key] = (
                    td_init[key][bs].clone().unsqueeze(0)
                )  # Make it shape (1,19) instead (19) for unsplited_task_length
            else:
                small_td[key][0] = td_init[key][bs]

        small_td_init = small_env.reset(small_td)

        start_time = time.time()
        out = model(
            small_td_init, phase="test", decode_type="greedy", return_actions=True
        )
        end_time = time.time()

        runtimes.append(end_time - start_time)
        mission_times.append(-out["reward"][0])
        actions.append(out["actions"][0])

    mission_times = [elem.cpu().detach().numpy() for elem in mission_times]
    mission_times = np.array(mission_times)
    runtimes = np.array(runtimes)

    print("\n\n\n------------------------------------------")
    mean = np.mean(mission_times)
    sd = np.std(mission_times)
    se = sd / np.sqrt(mission_times.shape[0])
    print(f"Avg Mission Time(s): {mean} ± {sd}. \nStandard Error: {se}")

    avg_runtime = np.mean(runtimes)
    print(f"Avg Wall Time(s): {avg_runtime}")
    print("------------------------------------------\n\n\n")

    # Save results to csv
    num_data = mission_times.shape[0]
    actions = [elem.cpu().detach().tolist() for elem in actions]
    results_dict = {
        "data_path": np.repeat(data_path, num_data),
        "mtsp_problem_type": np.repeat(args.method_mtsp_problem_type, num_data),
        "exp_idx": np.arange(num_data),
        "mission_time": mission_times,
        "tour": actions,
        "runtime": runtimes,
        "model_seed": np.repeat(args.method_seed, num_data),
    }
    df = pd.DataFrame(results_dict)
    df.to_csv(result_path)
    return


if __name__ == "__main__":
    main()
