import torch

from graph_attention_replanner.config import (
    LogFileConfig,
    get_generator,
    get_env,
)


"""
This script load the saved evaluation results and return details of it to plotting scripts.

Usage:
    env, td = get_gat_details(args)
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


def get_gat_details(args):
    # args = MissionPlanConfig().parse_eval_args()
    args.num_node_incl_home = args.num_node + 1

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

    data_path = data_cfg.get_data_logfilename(
        seed=args.seed, format="npz", override_mtsp_problem_type=args.mtsp_problem_type
    )  # Use Problem 5 (largest) data, and trim it down to smaller problem data))

    # Set up device
    device = torch.device(
        f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
    )

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
    if args.method_mtsp_problem_type <= 4:
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
        ]
        data_td["task_length"][:, : (1 + args.num_task)] = data_td[
            "unsplited_task_length"
        ][:, : (1 + args.num_task)]
        data_td["task_length"][:, (1 + args.num_task) :] = torch.zeros(
            args.max_num_agent
        )

    if args.method_mtsp_problem_type != args.mtsp_problem_type:
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

    else:
        method_env = data_env
        td_init = data_env.reset(data_td)

    return method_env, td_init
