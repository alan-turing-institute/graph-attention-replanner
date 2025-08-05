import os
import re
import glob
PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# Constants for setup WandB project name
WANDB_PROJECT_NAME = "GraphAttentionReplanner"

class LogFileConfig:
    def __init__(
        self,
        mtsp_problem_type,
        num_node,
        min_num_task,
        max_num_task,
        min_discretize_level,
        max_discretize_level,
        min_num_agent,
        max_num_agent,
        batch_size,
        read_only=False,  # when True, always return file path because it is for evaulation
    ):
        self.mtsp_problem_type = mtsp_problem_type  # There are 5 problem classes
        self.num_node = num_node
        self.min_num_task = min_num_task
        self.max_num_task = max_num_task
        self.min_discretize_level = min_discretize_level
        self.max_discretize_level = max_discretize_level
        self.min_num_agent = min_num_agent
        self.max_num_agent = max_num_agent
        self.batch_size = batch_size
        self.read_only = read_only
        self.wandb_project_name = WANDB_PROJECT_NAME
        self.cache_dir = os.path.expanduser(PROJECT_DIR + "/cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    @staticmethod
    def get_lkh_exec_path():
        return os.path.expanduser(PROJECT_DIR + "/external/LKH-3.0.10/LKH")

    def get_data_logfilename(
        self, seed=0, format="npz", override_mtsp_problem_type=None
    ):
        mtsp_problem_type = override_mtsp_problem_type or self.mtsp_problem_type
        path = "{}/data/general/problem{}_node{}_task{}to{}_dislevel{}to{}_agent{}to{}_seed{}_bs{}.{}".format(
            self.cache_dir,
            mtsp_problem_type,
            self.num_node,
            self.min_num_task,
            self.max_num_task,
            self.min_discretize_level,
            self.max_discretize_level,
            self.min_num_agent,
            self.max_num_agent,
            seed,
            self.batch_size,
            format,
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def get_data_lkhlogfiledir(self, seed=None, num_exp=1):
        path = "{}/data/lkh/problem{}_node{}_task{}to{}_dislevel{}to{}_agent{}to{}_seed{}_bs{}_exp{}".format(
            self.cache_dir,
            self.mtsp_problem_type,
            self.num_node,
            self.min_num_task,
            self.max_num_task,
            self.min_discretize_level,
            self.max_discretize_level,
            self.min_num_agent,
            self.max_num_agent,
            seed,
            self.batch_size,
            num_exp,
        )
        os.makedirs(path, exist_ok=True)
        return path

    def get_permu_logfiledir(self, seed=0):
        path = "{}/data/permu/problem{}_node{}_task{}to{}_dislevel{}to{}_agent{}to{}_bs{}_dataseed{}/".format(
            self.cache_dir,
            self.mtsp_problem_type,
            self.num_node,
            self.min_num_task,
            self.max_num_task,
            self.min_discretize_level,
            self.max_discretize_level,
            self.min_num_agent,
            self.max_num_agent,
            self.batch_size,
            seed,
        )
        os.makedirs(path, exist_ok=True)
        return path

    def get_result_logfiledir(self, method="enum", num_exp=1, seed=0):
        path = "{}/results/{}/problem{}_node{}_task{}to{}_dislevel{}to{}_agent{}to{}_bs{}_exp{}_dataseed{}/".format(
            self.cache_dir,
            method,
            self.mtsp_problem_type,
            self.num_node,
            self.min_num_task,
            self.max_num_task,
            self.min_discretize_level,
            self.max_discretize_level,
            self.min_num_agent,
            self.max_num_agent,
            self.batch_size,
            num_exp,
            seed,
        )
        os.makedirs(path, exist_ok=True)
        return path

    def get_result_logfilename(
        self,
        method="enum",
        num_exp=1,
        seed=0,
        method_seed=None,
        format="csv",
        override_mtsp_problem_type=None,
    ):
        mtsp_problem_type = override_mtsp_problem_type or self.mtsp_problem_type
        path = "{}/results/{}/problem{}_node{}_task{}to{}_dislevel{}to{}_agent{}to{}_bs{}_exp{}_dataseed{}".format(
            self.cache_dir,
            method,
            mtsp_problem_type,
            self.num_node,
            self.min_num_task,
            self.max_num_task,
            self.min_discretize_level,
            self.max_discretize_level,
            self.min_num_agent,
            self.max_num_agent,
            self.batch_size,
            num_exp,
            seed,
        )

        if method_seed is not None:
            # Add model seed for GAT models
            path += f"_methodseed{method_seed}"

        path += f".{format}"

        if self.read_only:
            return path
        if os.path.isfile(path):
            print(f"WARNING: Result file already exists: {path}. Skipping evaluation.")
            raise FileExistsError
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def get_train_logfiledir(self, seed=None, timestr="", algo="reinforce"):
        path = "{}/checkpoints/algo_{}/problem{}_node{}_task{}to{}_dislevel{}to{}_agent{}to{}_seed{}_{}".format(
            self.cache_dir,
            algo,
            self.mtsp_problem_type,
            self.num_node,
            self.min_num_task,
            self.max_num_task,
            self.min_discretize_level,
            self.max_discretize_level,
            self.min_num_agent,
            self.max_num_agent,
            seed,
            timestr,
        )
        os.makedirs(path, exist_ok=True)
        return path

    def get_train_modelpath(self, seed=None, policy="am", algo="reinforce"):
        prefix = "{}/checkpoints/algo_{}/problem{}_node{}_task{}to{}_dislevel{}to{}_agent{}to{}_seed{}*".format(
            self.cache_dir,
            algo,
            self.mtsp_problem_type,
            self.num_node,
            self.min_num_task,
            self.max_num_task,
            self.min_discretize_level,
            self.max_discretize_level,
            self.min_num_agent,
            self.max_num_agent,
            seed,
        )
        model_dirs = glob.glob(prefix)
        if len(model_dirs) > 1:
            print(
                f"WARNING: Multiple model directories found for prefix: {prefix}, Selecting the first one."
            )
            print(model_dirs)

        try:
            model_dir = model_dirs[0]
        except IndexError:
            print(f"ERROR: Cannot find any model at {prefix}")
            raise

        pattern = r"epoch_epoch=(\d+)\.ckpt"
        max_epoch = -1
        latest_file = None
        # Iterate through files in directory
        for filename in os.listdir(model_dir):
            match = re.match(pattern, filename)
            if match:
                epoch_num = int(match.group(1))
                if epoch_num > max_epoch:
                    max_epoch = epoch_num
                    latest_file = filename

        return f"{model_dir}/{latest_file}"