import argparse


class MissionPlanConfig:
    # Argument Parser
    def base_args(self):
        parser = argparse.ArgumentParser(
            description="MTSP Environment and Generator Configuration"
        )
        parser.add_argument(
            "--mtsp_problem_type",
            type=int,
            default=5,
            help="Type of MTSP problem to generate",
        )
        parser.add_argument(
            "--num_node",
            type=int,
            default=20,
            help="Number of node (excluding home)",
        )
        parser.add_argument(
            "--exact_num_node",
            action="store_true",
            default=False,
            help="Enable override num loc, so it is no longer up to, but exactly the number of node stated",
        )
        parser.add_argument(
            "--num_task",
            type=int,
            default=None,
            help="Number of (uniquely located) tasks",
        )
        parser.add_argument(
            "--min_num_task",
            type=int,
            default=20,
            help="Number of (uniquely located) tasks",
        )
        parser.add_argument(
            "--max_num_task",
            type=int,
            default=20,
            help="Number of (uniquely located) tasks",
        )
        parser.add_argument(
            "--discretize_level",
            type=int,
            default=None,
            help="Split 1 task into how many nodes",
        )
        parser.add_argument(
            "--min_discretize_level",
            type=int,
            default=1,
            help="Split 1 task into how many nodes",
        )
        parser.add_argument(
            "--max_discretize_level",
            type=int,
            default=2,
            help="Split 1 task into how many nodes",
        )
        parser.add_argument(
            "--num_agent", type=int, default=None, help="Minimum number of agents"
        )
        parser.add_argument(
            "--min_num_agent", type=int, default=4, help="Minimum number of agents"
        )
        parser.add_argument(
            "--max_num_agent", type=int, default=4, help="Maximum number of agents"
        )
        parser.add_argument("--batch_size", type=int, default=1000, help="Batch size")
        parser.add_argument(
            "--device_id", type=int, default=0, help="Device ID for computation"
        )
        parser.add_argument("--seed", type=int, default=0, help="Random seed")
        parser.add_argument(
            "--num_exp", type=int, default=1, help="Number of experiments"
        )
        return parser

    def process_minmax_args(self, args):
        if args.num_agent:
            print(
                f"WARNING: Number of agent is set to {args.num_agent}. Igoring min_num_agent and max_num_agent."
            )
            args.min_num_agent = args.num_agent
            args.max_num_agent = args.num_agent

        if args.num_task:
            print(
                f"WARNING: Number of task is set to {args.num_task}. Igoring min_num_task and max_num_task."
            )
            args.min_num_task = args.num_task
            args.max_num_task = args.num_task

        if args.discretize_level:
            print(
                f"WARNING: Discretize level is set to {args.discretize_level}. Igoring min_discretize_level and max_discretize_level."
            )
            args.min_discretize_level = args.discretize_level
            args.max_discretize_level = args.discretize_level
        return args

    def parse_args(self):
        parser = self.base_args()
        args = parser.parse_args()
        args = self.process_minmax_args(args)
        return args

    def parse_train_args(self):
        parser = self.base_args()
        parser.add_argument(
            "--embed_dim", type=int, default=128, help="Embedding dimension"
        )
        parser.add_argument(
            "--num_encoder_layers", type=int, default=8, help="Number of encoder layers"
        )
        parser.add_argument(
            "--num_heads", type=int, default=4, help="Number of attention heads"
        )
        parser.add_argument(
            "--train_data_size", type=int, default=1_000_000, help="Training data size"
        )
        parser.add_argument(
            "--val_data_size", type=int, default=100_000, help="Validation data size"
        )
        parser.add_argument(
            "--learning_rate", type=float, default=1e-5, help="Learning rate"
        )
        parser.add_argument(
            "--max_epochs", type=int, default=50, help="Maximum number of epochs"
        )

        parser.add_argument(
            "--algo",
            type=str,
            default="reinforce",
            help="Which RL algorithm to use. (reinforce/ppo/pomo/mreinforce/mppo/mpomo)",
        )

        parser.add_argument(
            "--disable_wandb",
            action="store_true",
            default=False,
            help="Disable wandb for debugging.",
        )

        args = parser.parse_args()
        args = self.process_minmax_args(args)
        return args

    def parse_eval_args(self):
        parser = self.base_args()
        parser.add_argument(
            "--method_mtsp_problem_type",
            type=int,
            default=5,
            help="Type of MTSP problem to the method is solving",
        )

        parser.add_argument(
            "--method_seed", type=int, default=0, help="Trained Model Seed"
        )

        args = parser.parse_args()
        args = self.process_minmax_args(args)
        return args

    def parse_eval_args_notebook(self):
        parser = self.base_args()
        parser.add_argument(
            "--method_mtsp_problem_type",
            type=int,
            default=5,
            help="Type of MTSP problem to the method is solving",
        )

        parser.add_argument(
            "--method_seed", type=int, default=0, help="Trained Model Seed"
        )

        args = parser.parse_args("")
        args = self.process_minmax_args(args)
        return args
