import csv
import numpy as np

from graph_attention_replanner.config import (
    MissionPlanConfig,
    LogFileConfig,
)
from graph_attention_replanner.method.lkh.lkh_mtsp_a_algorithm import solve

"""
This script runs and evaluates the LKH algorithm for solving MTSP problems.
Always put the spec of problem 5 as args, then based on the actual problem type we want to solve, we change mtsp_problem_type, this ensure we load the right data file.

python eval_lkh.py 
# method param
--method_mtsp_problem_type 1
# data param
--mtsp_problem_type 5 --num_node 12 --num_task 4 --discretize_level 3 --num_agent 3 --exact_num_node --batch_size 1000 --num_exp 1

python eval_lkh.py --method_mtsp_problem_type 1 --mtsp_problem_type 5 --num_node 12 --num_task 4 --discretize_level 3 --num_agent 3 --exact_num_node --batch_size 1000 --num_exp 1
"""


def main():
    args = MissionPlanConfig().parse_eval_args()
    args.num_node_incl_home = args.num_node + 1

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
    datadir = cfg.get_data_lkhlogfiledir(seed=args.seed, num_exp=args.num_exp)
    lkh_path = cfg.get_lkh_exec_path()

    try:
        results_path = cfg.get_result_logfilename(
            method="lkh",
            num_exp=args.num_exp,
            format="csv",
            override_mtsp_problem_type=args.method_mtsp_problem_type,
            seed=args.seed,
        )
    except FileExistsError:
        return
    with open(results_path, "w") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "data_path",
                "mtsp_problem_type",
                "exp_idx",
                "mission_time",
                "tour",
                "runtime",
            ]
        )

    mission_time_arr, runtime_arr = [], []
    for exp_idx in range(args.num_exp):
        extension = "tsp" if args.method_mtsp_problem_type == 1 else "atsp"
        filepath = datadir + f"/{exp_idx}.{extension}"
        tour, mission_time, runtime = solve(
            mtsp_problem_type=args.method_mtsp_problem_type,
            filepath=filepath,
            salesmen=args.num_agent,
            solver_path=lkh_path,
        )

        with open(results_path, "a") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    str(filepath),
                    args.method_mtsp_problem_type,
                    exp_idx,
                    mission_time,
                    tour,
                    runtime,
                ]
            )

        mission_time_arr.append(mission_time)
        runtime_arr.append(runtime)
        # df = pd.read_csv(results_path)
        # l = literal_eval(df["tour"].iloc[0])
        # print(l)

    print("\n\n\n------------------------------------------")
    avg_mission_time = np.mean(mission_time_arr)
    avg_run_time = np.mean(runtime_arr)
    print(f"Avg Mission Time(s): {avg_mission_time}\nAvg Run Time(s): {avg_run_time}")
    print("------------------------------------------\n\n\n")


if __name__ == "__main__":
    main()
