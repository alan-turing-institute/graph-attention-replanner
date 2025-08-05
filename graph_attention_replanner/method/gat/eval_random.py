import torch
import numpy as np
import pandas as pd
import time

from graph_attention_replanner.config import (
    MissionPlanConfig,
    LogFileConfig,
    get_generator,
    get_env,
)

"""
This script shows the random benchmark. 

For help:
python eval_random.py --help

Example Command:
python eval_gat.py --method_seed 0 --seed 0 --method_mtsp_problem_type 4 --mtsp_problem_type 4 --num_node 12 --num_task 4 --discretize_level 1 --num_agent 3 --batch_size 100 --num_exp 100
"""


def generate_random_mission_plans(
    batch_size=None,
    num_agent=None,
    num_node_incl_home=None,
    padding=None,  # Number of extra column to apply
    method_mtsp_problem_type=None,
):
    if method_mtsp_problem_type < 3:
        zeros = np.zeros((batch_size, num_agent - 1))
        nodes = np.arange(1, num_node_incl_home)
        nodes = np.tile(nodes, (batch_size, 1))
        plans = np.concatenate((nodes, zeros), axis=1)
        shuffled_plans = np.apply_along_axis(np.random.permutation, axis=1, arr=plans)

    if method_mtsp_problem_type >= 3:  # stick start node after 0
        num_node_incl_home = num_node_incl_home - num_agent
        zeros = np.zeros((batch_size, num_agent - 1))
        nodes = np.arange(1, num_node_incl_home)
        nodes = np.tile(nodes, (batch_size, 1))
        plans = np.concatenate((nodes, zeros), axis=1)
        shuffled_plans = np.apply_along_axis(np.random.permutation, axis=1, arr=plans)

        zero = np.zeros((batch_size, 1))
        shuffled_plans = np.concatenate(
            (zero, shuffled_plans), axis=1
        )  # Add a column of zero at the start
        start_loc = np.arange(num_node_incl_home, num_node_incl_home + num_agent)
        # print("Before Plan", shuffled_plans[0])
        new_plans = []
        for plan_idx, _ in enumerate(shuffled_plans):
            indices = np.where(shuffled_plans[plan_idx] == 0)
            # indices = np.add(indices, 1) # cannot use this cannot insert at idx 2 if arr is of len 3 like [0,0,0]
            indices = indices[0]  # get the first element of array
            PLACEHOLDER = -1
            plan = np.insert(
                shuffled_plans[plan_idx], indices.flatten().tolist(), PLACEHOLDER
            )
            # now we have [-1, 0 ... -1, 0 ... -1, 0]
            plan[plan == 0] = start_loc  # replace all zero as start loc
            plan[plan == PLACEHOLDER] = 0  # replace all placeholder (-1) as zero
            # indices = np.where(plan == PLACEHOLDER)
            # plan[indices] = start_loc
            new_plans.append(plan)
        shuffled_plans = np.array(new_plans)
        # print("After Plan", shuffled_plans[0])
        # _ = input("Press Enter to continue")

        shuffled_plans = np.delete(
            shuffled_plans, 0, 1
        )  # delete the starting zero column

    if padding is not None:
        # TODO: Not tested
        nodes = np.arange(num_node_incl_home, num_node_incl_home + padding)
        nodes = np.tile(nodes, (batch_size, 1))
        shuffled_plans = np.concatenate((shuffled_plans, nodes), axis=1)

    return shuffled_plans


def main():
    args = MissionPlanConfig().parse_eval_args()
    np.random.seed(args.method_seed)

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
            method="random",
            format="csv",
            override_mtsp_problem_type=args.method_mtsp_problem_type,
            seed=args.seed,
        )
    except FileExistsError:
        return
    data_path = data_cfg.get_data_logfilename(
        seed=args.seed, format="npz", override_mtsp_problem_type=args.mtsp_problem_type
    )  # Use Problem 5 (largest) data, and trim it down to smaller problem data))
    print(
        f"Evaluating using data from {data_path} and saving results at {result_path} ..."
    )

    # Set up device
    device = torch.device(
        f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
    )

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

    if data_td["start_locs"] is not None:
        print(data_td["start_locs"][0])
    print(f"------------Loaded Dataset from {data_path}-------------")

    # Set generator and env based on the problem type we want to solve for.
    if args.method_mtsp_problem_type <= 4:
        method_num_node = args.num_task
        method_discretize_level = 1
    else:
        # TODO: Might not work for phase 2 experiments
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

    sampled_population = 501  # odd number
    sampled_mission_time = []
    sampled_actions = []
    start_time = time.time()
    for _ in range(sampled_population):
        actions_np = generate_random_mission_plans(
            batch_size=td_init["locs"].shape[0],
            num_agent=args.max_num_agent,
            num_node_incl_home=data_td["locs"].shape[
                1
            ],  # TODO: change this for generalised model
            method_mtsp_problem_type=args.method_mtsp_problem_type,
        )
        action = torch.from_numpy(actions_np).type(torch.int64).to(device)
        mission_time = method_env.get_reward_with_new_locs(td=td_init, actions=action)
        sampled_mission_time.append(mission_time)
        sampled_actions.append(action)

    mission_times, actions = get_median(sampled_mission_time, sampled_actions)
    end_time = time.time()
    execution_time = end_time - start_time

    print("\n\n\n------------------------------------------")
    # sd, mean = torch.std_mean(mission_times)
    # se = sd / np.sqrt(mission_times.size(dim=0))
    mean = np.mean(mission_times)
    sd = np.std(mission_times)
    se = sd / np.sqrt(mission_time.shape[0])
    print(f"Avg Mission Time(s): {mean} ± {sd}. \nStandard Error: {se}")
    avg_runtime = execution_time / args.batch_size
    print(f"Avg Wall Time(s): {avg_runtime}")
    print("------------------------------------------\n\n\n")

    # Save results to csv
    num_data = mission_times.shape[0]
    results_dict = {
        "data_path": np.repeat(data_path, num_data),
        "mtsp_problem_type": np.repeat(args.method_mtsp_problem_type, num_data),
        "exp_idx": np.arange(num_data),
        "mission_time": mission_times.tolist(),  # must be to list to save it properly
        # "tour": actions.tolist(),  # must be to list to save it properly
        "tour": actions,  # must be to list to save it properly, already a list
        "runtime": np.repeat(avg_runtime, num_data),
    }
    df = pd.DataFrame(results_dict)
    df.to_csv(result_path)

    return


def get_median(time, tour):
    time = np.array(time)
    median_time = np.median(time[:, :], axis=0)
    median_indices = np.abs(time - median_time).argmin(axis=0)

    # print(time[:, 0])
    # print(median_time[0])
    # print(median_indices[0])
    # swap 1st and second dimension
    median_tours = []
    swapped = [[tour[j][i] for j in range(len(tour))] for i in range(len(tour[0]))]
    for i, paths in enumerate(swapped):
        median_tours.append(paths[median_indices[i]].tolist())
    # print(swapped[0])
    # print(median_tours[0])

    return median_time, median_tours


if __name__ == "__main__":
    main()
