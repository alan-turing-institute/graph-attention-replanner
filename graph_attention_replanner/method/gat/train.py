import time
import wandb
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from rl4co.utils.trainer import RL4COTrainer
from rl4co.models import (
    AttentionModelPolicy,
    REINFORCE,
    AMPPO,
    POMO,
    MVMoE_AM,
    MVMoE_POMO,
)

from graph_attention_replanner.config import (
    MissionPlanConfig,
    LogFileConfig,
    get_generator,
    get_env,
)


"""
This script trains GATR models.

Example Command:
# Quick Run for debugging:
python train.py --num_task 4 --num_agent 3 --discretize_level 3 --num_node 12 --mtsp_problem_type 5 --batch_size 10 --train_data_size 1000 --val_data_size 100 --max_epochs 1 --disable_wandb

# For training:
python train.py --num_task 4 --num_agent 3 --discretize_level 3 --num_node 12  --mtsp_problem_type 1
"""


def main():
    args = MissionPlanConfig().parse_train_args()

    print(vars(args))
    timestr = time.strftime("%Y%m%d%H%M%S")

    # Device Selection
    device = torch.device(
        f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
    )
    L.seed_everything(args.seed, workers=True)

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
    env = get_env(args.mtsp_problem_type)(generator, device=device)

    # Debug
    # td_data = env.generator(args.batch_size)
    # print(td_data)

    logfileconfig = LogFileConfig(
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

    checkpoint_dir = logfileconfig.get_train_logfiledir(seed=args.seed, timestr=timestr, algo=args.algo)

    if not args.disable_wandb:
        wandb.login()
        wandb_name = checkpoint_dir.split("/")[-1]
        logger = WandbLogger(
            project=logfileconfig.wandb_project_name, name=wandb_name, config=vars(args)
        )

    policy_kwargs = {
        "embed_dim": args.embed_dim,
        "num_encoder_layers": args.num_encoder_layers,
        "num_heads": args.num_heads,
    }
    if args.algo == "reinforce":
        policy = AttentionModelPolicy(env_name=env.name, **policy_kwargs)
        model = REINFORCE(
            env,
            policy,
            baseline="rollout",
            batch_size=args.batch_size,
            train_data_size=args.train_data_size,
            val_data_size=args.val_data_size,
            optimizer_kwargs={"lr": args.learning_rate},
        )
    elif args.algo == "ppo":
        model = AMPPO(
            env,
            policy=None,
            policy_kwargs=policy_kwargs,
            batch_size=args.batch_size,
            train_data_size=args.train_data_size,
            val_data_size=args.val_data_size,
            optimizer_kwargs={"lr": args.learning_rate},
        )
    elif args.algo == "pomo":
        model = POMO(
            env,
            policy=None,
            policy_kwargs=policy_kwargs,
            batch_size=args.batch_size,
            train_data_size=args.train_data_size,
            val_data_size=args.val_data_size,
            optimizer_kwargs={"lr": args.learning_rate},
        )
    elif args.algo == "mreinforce":
        model = MVMoE_AM(
            env,
            policy=None,
            policy_kwargs=policy_kwargs,
            batch_size=args.batch_size,
            train_data_size=args.train_data_size,
            val_data_size=args.val_data_size,
            optimizer_kwargs={"lr": args.learning_rate},
        )
    elif args.algo == "mpomo":
        model = MVMoE_POMO(
            env,
            policy=None,
            policy_kwargs=policy_kwargs,
            batch_size=args.batch_size,
            train_data_size=args.train_data_size,
            val_data_size=args.val_data_size,
            optimizer_kwargs={"lr": args.learning_rate},
        )
    else:
        raise ValueError(f"Invalid algorithm: {args.algo}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch_{epoch:03d}",
        save_top_k=1,
        save_last=False,
        monitor="val/reward",
        mode="max",
    )
    early_stop_callback = EarlyStopping(
        monitor="val/reward", min_delta=0.00, patience=3, verbose=False, mode="max"
    )
    rich_model_summary = RichModelSummary(max_depth=3)
    callbacks = [checkpoint_callback, rich_model_summary, early_stop_callback]

    if not args.disable_wandb:
        trainer = RL4COTrainer(
            accelerator="gpu",
            max_epochs=args.max_epochs,
            logger=logger,
            callbacks=callbacks,
        )
    else:
        trainer = RL4COTrainer(
            accelerator="gpu",
            max_epochs=args.max_epochs,
            callbacks=callbacks,
        )

    trainer.fit(model)
    if not args.disable_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
