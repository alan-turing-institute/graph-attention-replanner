import torch
import numpy as np

from graph_attention_replanner.config import MissionPlanConfig, get_generator, get_env

"""
This script loops through the environment manually, user can manually set the actions.

Example Command:
python run_one_sample.py --num_task 4 --num_agent 3 --discretize_level 1 --num_node 4 --batch_size 1 --mtsp_problem_type 3
"""


def main():
    args = MissionPlanConfig().parse_args()

    # Initialize Generator and Environment
    generator = get_generator(args.mtsp_problem_type)(
        num_node=args.num_node,
        min_num_task=args.min_num_task,
        max_num_task=args.max_num_task,
        min_discretize_level=args.min_discretize_level,
        max_discretize_level=args.max_discretize_level,
        min_num_agent=args.min_num_agent,
        max_num_agent=args.max_num_agent,
        seed=args.seed,
    )
    env = get_env(args.mtsp_problem_type)(generator)
    td_data = env.generator(args.batch_size)  # The initial batch data goes here.
    td = env.reset(td_data)
    print("[DEBUG] Cost", td["cost_matrix"])
    print("[DEBUG] td avaliable:", td["action_mask"])

    print(
        "========================== Selecting Start Location A =========================="
    )
    action = torch.tensor(np.array([[5], [5]]))
    td.set("action", action)
    print(f"[DEBUG] -----action: {action}")
    td = env.step(td)["next"]

    print(
        "========================== Agent 0 Selecting Task 1 =========================="
    )
    action = torch.tensor(np.array([[1], [1]]))
    td.set("action", action)
    print(f"[DEBUG] -----action: {action}")
    td = env.step(td)["next"]

    print("========================== Selecting Depot ==========================")
    action = torch.tensor(np.array([[0], [0]]))
    td.set("action", action)
    print(f"[DEBUG] -----action: {action}")
    td = env.step(td)["next"]

    print(
        "========================== Selecting Start Location B =========================="
    )
    action = torch.tensor(np.array([[6], [6]]))
    td.set("action", action)
    print(f"[DEBUG] -----action: {action}")
    td = env.step(td)["next"]

    print("========================== Selecting Depot ==========================")
    action = torch.tensor(np.array([[0], [0]]))
    td.set("action", action)
    print(f"[DEBUG] -----action: {action}")
    td = env.step(td)["next"]

    print(
        "========================== Selecting Start Location C =========================="
    )
    action = torch.tensor(np.array([[7], [7]]))
    td.set("action", action)
    print(f"[DEBUG] -----action: {action}")
    td = env.step(td)["next"]

    print(
        "========================== Agent 2 Selecting Task 2. Expect Depot and Task 1 and all Start Loc are not avaliable =========================="
    )
    action = torch.tensor(np.array([[2], [2]]))
    td.set("action", action)
    print(f"[DEBUG] -----action: {action}")
    td = env.step(td)["next"]

    print(
        "========================== Agent 2 Selecting Task 3 =========================="
    )
    action = torch.tensor(np.array([[3], [3]]))
    td.set("action", action)
    print(f"[DEBUG] -----action: {action}")
    td = env.step(td)["next"]

    print(
        "========================== Agent 2 Selecting Task 4 =========================="
    )
    action = torch.tensor(np.array([[4], [4]]))
    td.set("action", action)
    print(f"[DEBUG] -----action: {action}")
    td = env.step(td)["next"]

    print("========================== Selecting Depot ==========================")
    action = torch.tensor(np.array([[0], [0]]))
    td.set("action", action)
    print(f"[DEBUG] -----action: {action}")
    td = env.step(td)["next"]

    print("========================== Selecting Depot ==========================")
    action = torch.tensor(np.array([[0], [0]]))
    td.set("action", action)
    print(f"[DEBUG] -----action: {action}")
    td = env.step(td)["next"]

    print("========================== Selecting Depot ==========================")
    action = torch.tensor(np.array([[0], [0]]))
    td.set("action", action)
    print(f"[DEBUG] -----action: {action}")
    td = env.step(td)["next"]


if __name__ == "__main__":
    main()
