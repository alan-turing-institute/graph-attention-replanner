import numpy as np
import os
from ast import literal_eval
import pandas as pd

# from tqdm import tqdm
import dask.dataframe as dd
import csv
import torch

from graph_attention_replanner.config import (
    MissionPlanConfig,
    LogFileConfig,
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_max(x):
    arr = torch.tensor(literal_eval(x)).to(DEVICE)
    return torch.max(arr)


def get_hist_param(x, max_val, bin_width):
    arr = torch.tensor(literal_eval(x)).to(DEVICE)
    bins = torch.arange(0, max_val + bin_width, bin_width).to(DEVICE)
    return torch.histogram(arr, bins=bins)


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
        method="enum", num_exp=args.num_exp, format="csv"
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
        hist_results_file = cfg.get_result_logfilename(
            method="enumhist", num_exp=args.num_exp, format="csv"
        )
    except FileExistsError:
        return

    # Getting the range of the histogram
    df_dask = dd.read_csv(enum_results_file)
    df_max = (
        df_dask["all_mission_time"]
        .apply(get_max, meta=pd.Series(dtype="float64"))
        .compute()
    )
    max_val = df_max.max()
    bin_width = 0.1

    # Getting the histogram count
    df_meta = pd.DataFrame(columns=["hist", "bin"], dtype="float64")
    df = (
        df_dask["all_mission_time"]
        .apply(lambda x: get_hist_param(x, max_val, bin_width), meta=df_meta)
        .compute()
    )  # convert to pandas

    hists = []
    for index, row in df.items():
        hists.append(row[0])
    final_hist = np.sum(hists, axis=0)

    with open(hist_results_file, "w") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Hists",
                "Max Val",
                "Bin width",
            ]
        )
        writer.writerow(
            [
                final_hist.tolist(),
                float(max_val),
                bin_width,
            ]
        )


if __name__ == "__main__":
    main()
