from graph_attention_replanner.config import (
    MissionPlanConfig,
    LogFileConfig,
    get_generator,
    get_env,
)


"""
This script generates data for the 5 different MTSP problem types.

For help:
python gen_data.py --help

Example Command:
# Create a dataset of 1 data instance with the problem setup (4 nodes, 4 tasks, 1 discretization level, 3 agents)
python gen_data.py --num_node 4 --num_task 4 --discretize_level 1 --num_agent 3 --batch_size 1

# Create a dataset of 10 data instances with the problem setup (10 nodes, 4-5 tasks, 1-2 discretization levels, 3 agents)
python gen_data.py --num_node 10 --min_num_task 4 --max_num_task 5 --min_discretize_level 1 --max_discretize_level 2 --num_agent 3 --batch_size 10
"""


def main():
    args = MissionPlanConfig().parse_args()

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

    # Save data
    filename = LogFileConfig(
        args.mtsp_problem_type,
        args.num_node,
        args.min_num_task,
        args.max_num_task,
        args.min_discretize_level,
        args.max_discretize_level,
        args.min_num_agent,
        args.max_num_agent,
        args.batch_size,
    ).get_data_logfilename(seed=args.seed, format="npz")
    env.generator.save_data(td_data, filename)

if __name__ == "__main__":
    main()
