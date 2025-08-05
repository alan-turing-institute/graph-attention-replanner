import os
import numpy as np
import time
import csv
import pandas as pd

from graph_attention_replanner.config import (
    MissionPlanConfig,
    LogFileConfig,
)
from graph_attention_replanner.method.enumeration.enumeration_breakattime import Enumeration


def main():
    args = MissionPlanConfig().parse_eval_args()
    args.num_node_incl_home = args.num_node + 1
    args.num_task_incl_home = args.num_task + 1

    cfg = LogFileConfig(
        args.method_mtsp_problem_type,
        args.num_node,
        args.min_num_task,
        args.max_num_task,
        args.min_discretize_level,
        args.max_discretize_level,
        args.min_num_agent,
        args.max_num_agent,
        args.batch_size,
        read_only=True,
    )
    data_path = cfg.get_data_logfilename(
        seed=args.seed, format="npz", override_mtsp_problem_type=args.mtsp_problem_type
    )  # Use Problem 5 (largest) data, and trim it down to smaller problem data)
    median_path = cfg.get_result_logfilename(
        method="median_timeonly",
        num_exp=args.num_exp,
        format="csv",
        seed=args.seed,
    )
    if not os.path.isfile(median_path):
        print(
            f"WARNING: Cannot do median time to tour conversion, {median_path} does not exist."
        )
        return
    median_df = pd.read_csv(median_path)

    data = np.load(data_path)
    # generator 4 doesnt have this key
    unsplited_task_length = (
        data["unsplited_task_length"]
        if "unsplited_task_length" in data
        else data["task_length"]
    )

    if args.method_mtsp_problem_type == 1:
        locs_for_algo = data["locs"][: args.num_exp, : args.num_task_incl_home, :]
        task_length_for_algo = np.zeros((args.num_exp, args.num_task_incl_home))
        home_depots = data["locs"][: args.num_exp, 0, :].reshape((args.num_exp, 1, 2))
        initial_drone_locations_for_algo = np.broadcast_to(
            home_depots, (args.num_exp, args.num_agent, 2)
        )
    if args.method_mtsp_problem_type == 2:
        locs_for_algo = data["locs"][: args.num_exp, : args.num_task_incl_home, :]
        task_length_for_algo = unsplited_task_length[
            : args.num_exp, : args.num_task_incl_home
        ]
        home_depots = data["locs"][: args.num_exp, 0, :].reshape((args.num_exp, 1, 2))
        initial_drone_locations_for_algo = np.broadcast_to(
            home_depots, (args.num_exp, args.num_agent, 2)
        )
    if args.method_mtsp_problem_type == 3:
        locs_for_algo = data["locs"][: args.num_exp, : args.num_task_incl_home, :]
        task_length_for_algo = np.zeros((args.num_exp, args.num_task_incl_home))
        initial_drone_locations_for_algo = data["start_locs"]
    if (
        args.method_mtsp_problem_type == 4
        or args.method_mtsp_problem_type == 6
        or args.method_mtsp_problem_type == 7
    ):
        locs_for_algo = data["locs"][: args.num_exp, : args.num_task_incl_home, :]
        task_length_for_algo = unsplited_task_length[
            : args.num_exp, : args.num_task_incl_home
        ]
        initial_drone_locations_for_algo = data["start_locs"]
    if args.method_mtsp_problem_type == 5:
        locs_for_algo = data["locs"][: args.num_exp, : args.num_node_incl_home, :]
        task_length_for_algo = data["task_length"][
            : args.num_exp, : args.num_node_incl_home
        ]
        initial_drone_locations_for_algo = data["start_locs"]

    num_data = locs_for_algo.shape[0]
    assert (
        args.num_exp <= num_data
    ), "Number of experiments {num_exp} must be less than the number of data points {num_data}."

    try:
        cfg = LogFileConfig(
            args.method_mtsp_problem_type,
            args.num_node,
            args.min_num_task,
            args.max_num_task,
            args.min_discretize_level,
            args.max_discretize_level,
            args.min_num_agent,
            args.max_num_agent,
            args.batch_size,
        )
        summary_log_file = cfg.get_result_logfilename(
            method="median", num_exp=args.num_exp, format="csv", seed=args.seed
        )
    except FileExistsError:
        return
    with open(summary_log_file, "w") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "data_path",
                "mtsp_problem_type",
                "exp_idx",
                "mission_time",
                "estimated_mission_time",
                "tour",
                "runtime",
            ]
        )
    permu_cache_dir = cfg.get_permu_logfiledir(seed=args.seed)

    mission_time_arr, runtime_arr = [], []
    for exp_idx in range(args.num_exp):
        target_mission_time = median_df[median_df["exp_idx"] == exp_idx][
            "mission_time"
        ].values[0]
        print("target_mission_time", target_mission_time)
        print(f"Running experiment {exp_idx+1}/{args.num_exp}...")
        task_times = task_length_for_algo[exp_idx][1:]
        initial_drone_locations = initial_drone_locations_for_algo[exp_idx]
        task_locations = locs_for_algo[exp_idx][1:]
        home_depot = locs_for_algo[exp_idx][0].reshape((1, 2))

        print("Initial Drone Locations: ", initial_drone_locations)
        print("Task Locations: ", task_locations)
        print("Home Depot: ", home_depot)
        print("Task Times: ", task_times)

        start_time = time.time()
        best_mission_time, the_mission_plan = Enumeration(
            initial_drone_locations,
            task_locations,
            task_times,
            home_depot,
            args.num_task,
            args.discretize_level,
            permu_cache_dir,
            target_time=target_mission_time,
        ).run()
        end_time = time.time()
        runtime = end_time - start_time

        with open(summary_log_file, "a") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    str(data_path),
                    args.method_mtsp_problem_type,
                    exp_idx,
                    target_mission_time,
                    best_mission_time,
                    list(the_mission_plan),
                    runtime,
                ]
            )
        mission_time_arr.append(best_mission_time)
        runtime_arr.append(runtime)

    print("\n\n\n------------------------------------------")
    avg_mission_time = np.mean(mission_time_arr)
    avg_run_time = np.mean(runtime_arr)
    print(f"Avg Mission Time(s): {avg_mission_time}\nAvg Run Time(s): {avg_run_time}")
    print("------------------------------------------\n\n\n")


if __name__ == "__main__":
    main()
