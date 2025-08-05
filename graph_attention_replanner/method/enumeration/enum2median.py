import os
from ast import literal_eval
import pandas as pd

# from tqdm import tqdm
import dask.dataframe as dd
import torch

from graph_attention_replanner.config import (
    MissionPlanConfig,
    LogFileConfig,
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def process_cell(x):
    arr = torch.tensor(literal_eval(x)).to(DEVICE)
    # if is even, pad an extra element at the front, ensure median is a real value that exist
    if arr.size(0) % 2 == 0:
        arr = torch.cat([torch.tensor([0]).to(DEVICE), arr])
    return float(torch.median(arr))


def main():
    args = MissionPlanConfig().parse_eval_args()
    args.num_node_incl_home = args.num_node + 1
    args.num_task_incl_home = args.num_task + 1

    cfg_read_only = LogFileConfig(
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
    enum_results_file = cfg_read_only.get_result_logfilename(
        method="enum", seed=args.seed, num_exp=args.num_exp, format="csv"
    )
    if not os.path.isfile(enum_results_file):
        print(
            f"WARNING: Cannot do enum to median conversion, {enum_results_file} does not exist."
        )
        return

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

    try:
        rand_results_file = cfg.get_result_logfilename(
            method="median_timeonly",
            seed=args.seed,
            num_exp=args.num_exp,
            format="csv",
        )
    except FileExistsError:
        return

    df = dd.read_csv(enum_results_file)
    df_random = df.loc[
        :, df.columns != "all_mission_time"
    ].copy()  # Copy all columns except all_mission_time cuz it is large

    df_random["mission_time"] = df["all_mission_time"].apply(
        process_cell, meta=pd.Series(dtype="float64")
    )

    # Convert dask dataframe to pandas, then
    df_random_pd = df_random.compute()
    df_random_pd.to_csv(rand_results_file)


if __name__ == "__main__":
    # tqdm.pandas()
    main()
