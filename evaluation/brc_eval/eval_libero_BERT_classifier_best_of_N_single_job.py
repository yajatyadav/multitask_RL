"""
BRC job worker for LIBERO BERT best-of-N evaluation.

Each invocation evaluates exactly one N value over one chunk of evaluation envs.
Outputs are intermediate per-job pickle files to be aggregated by postprocessing.
"""

import argparse
import copy
import json
import os
import pickle
import sys
import time
from pathlib import Path

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from agents.acbcflowactor import ACBCFlowActorAgent, get_config as get_actor_config
from envs.env_utils import make_env_and_datasets
from evaluation_libero import evaluate
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState

os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_triton_gemm_any=True"


class TransClassifier_BERT(nn.Module):
    vision_encoder: nn.Module = None
    layer_norm: bool = True
    p_drop_state: float = 0.5
    embed_dim: int = 128

    def setup(self):
        from utils.networks import MLP

        self.action_encoder = MLP((self.embed_dim, self.embed_dim), activate_final=True, layer_norm=self.layer_norm)
        self.lang_encoder = MLP((self.embed_dim, self.embed_dim), activate_final=True, layer_norm=self.layer_norm)
        self.classifier = MLP((self.embed_dim, self.embed_dim, 1), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, actions, language_embedding, train=True, rng=None):
        assert actions is not None, "Actions must be provided to the classifier"
        assert self.vision_encoder is not None, "Encoder must be provided to the classifier"
        obs_encoded = self.vision_encoder(observations)
        if rng is None:
            rng = jax.random.PRNGKey(0)
        mask = jax.random.bernoulli(rng, 1 - self.p_drop_state, shape=(obs_encoded.shape[0], 1))
        obs_encoded = jax.lax.cond(train, lambda x: mask * x, lambda x: x, obs_encoded)
        action_encoded = self.action_encoder(actions)
        lang_encoded = self.lang_encoder(language_embedding)
        inputs = jnp.concatenate([obs_encoded, action_encoded, lang_encoded], axis=-1)
        return self.classifier(inputs)


def load_classifier(save_dir, ckpt_num):
    save_dir = Path(save_dir)
    with open(save_dir / f"hparams_{ckpt_num}.json", "r", encoding="utf-8") as f:
        hparams = json.load(f)
    with open(save_dir / f"params_{ckpt_num}.pkl", "rb") as f:
        params = pickle.load(f)
    classifier_def = TransClassifier_BERT(
        vision_encoder=encoder_modules[hparams["encoder"]](),
        embed_dim=hparams["embed_dim"],
        layer_norm=hparams["layer_norm"],
        p_drop_state=hparams["p_drop_state"],
    )
    network_def = ModuleDict({"classifier": classifier_def})
    network_tx = optax.adam(learning_rate=hparams["lr"])
    return TrainState.create(network_def, params, tx=network_tx)


class ClassifierAgent:
    def __init__(
        self,
        classifier_network_restore_dir,
        classifier_ckpt_num,
        actor_restore_path,
        example_batch,
        horizon_length,
        actor_encoder,
        num_samples,
        actor_seed,
    ):
        self.classifier_network = load_classifier(classifier_network_restore_dir, classifier_ckpt_num)
        self.actor_network = self._restore_actor_network(
            actor_restore_path,
            copy.deepcopy(example_batch),
            horizon_length,
            actor_encoder,
            actor_seed,
        )
        self.config = {
            "action_dim": example_batch["actions"].shape[-1],
            "horizon_length": horizon_length,
            "action_chunking": True,
            "actor_encoder": actor_encoder,
            "flow_steps": 10,
            "num_samples": num_samples,
        }

    def _restore_actor_network(self, actor_restore_path, example_batch, horizon_length, actor_encoder, actor_seed):
        with open(actor_restore_path, "rb") as f:
            load_dict = pickle.load(f)
        actor_config = get_actor_config()
        actor_config["encoder"] = actor_encoder
        actor_config["horizon_length"] = horizon_length
        actor_agent = ACBCFlowActorAgent.create(
            seed=actor_seed,
            ex_observations=copy.deepcopy(example_batch["observations"]),
            ex_actions=copy.deepcopy(example_batch["actions"]),
            config=actor_config,
        )
        loaded_actor_params = load_dict["agent"]["network"]["params"]
        actor_agent_state_dict = flax.serialization.to_state_dict(actor_agent)
        actor_agent_params = actor_agent_state_dict["network"]["params"]
        for key in actor_agent_params:
            if "actor" in key:
                actor_agent_params[key] = loaded_actor_params[key]
        actor_agent = flax.serialization.from_state_dict(actor_agent, actor_agent_state_dict)
        return actor_agent.network

    def sample_actions(self, observations, rng=None, temperature=1.0, print_debug=False):
        del temperature, print_debug
        full_action_dim = self.config["action_dim"] * self.config["horizon_length"]
        batch_size = observations[sorted(observations.keys())[0]].shape[0]
        lang_embedding = observations.get("language")

        noises = jax.random.normal(rng, (batch_size, self.config["num_samples"], full_action_dim))
        observations = jax.tree_util.tree_map(
            lambda x: jnp.repeat(x[:, None, ...], self.config["num_samples"], axis=1),
            observations,
        )
        actions = self._compute_flow_actions(observations, noises)
        actions = jnp.clip(actions, -1, 1)

        observations.pop("proprio")
        flat_obs = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), observations)
        flat_actions = jnp.reshape(actions, (-1, full_action_dim))
        flat_lang = jnp.repeat(lang_embedding, self.config["num_samples"], axis=0)
        logits = self.classifier_network.select("classifier")(flat_obs, flat_actions, flat_lang, train=False)
        logits = jnp.reshape(logits, (batch_size, self.config["num_samples"]))
        indices = jnp.argmax(logits, axis=-1)
        return actions[jnp.arange(batch_size), indices, :]

    def _compute_flow_actions(self, observations, noises):
        if self.config["actor_encoder"] is not None:
            observations = self.actor_network.select("actor_flow_encoder")(observations)
        actions = noises
        for i in range(self.config["flow_steps"]):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config["flow_steps"])
            vels = self.actor_network.select("actor_flow")(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config["flow_steps"]
        return jnp.clip(actions, -1, 1)


def _validate_chunk_args(chunk_id, num_chunks):
    if num_chunks < 1:
        raise ValueError("num_env_chunks must be >= 1")
    if chunk_id < 0 or chunk_id >= num_chunks:
        raise ValueError("env_chunk_id must be in [0, num_env_chunks)")


def run_single_job(args):
    run_root = os.path.join(args.intermediate_base_dir, args.wandb_run_name)
    os.makedirs(run_root, exist_ok=True)
    _validate_chunk_args(args.env_chunk_id, args.num_env_chunks)

    _, eval_envs_iterator, dataset, _ = make_env_and_datasets(
        args.env_name,
        args.task_name,
        args.language_embedder,
        augmentation_type="none",
        augmentation_reward=False,
        num_parallel_envs=args.num_parallel_envs,
        keys_to_load=args.keys_to_load,
        batch_level_sampling=True,
        demo_nums_to_use_per_task=args.demo_nums_to_use_per_task,
        augmentation_dict=None,
        is_notebook=False,
    )
    example_batch = dataset.sample_sequence(1, sequence_length=args.horizon_length, discount=0.99)
    agent = ClassifierAgent(
        args.classifier_restore_dir,
        args.classifier_ckpt_num,
        args.actor_restore_path,
        example_batch,
        args.horizon_length,
        args.actor_encoder,
        args.n_val,
        args.actor_seed,
    )

    selected_env_results = []
    all_env_names = []
    selected_indices = []
    total_envs = 0
    for idx, (env_obj, env_name) in enumerate(eval_envs_iterator):
        total_envs += 1
        all_env_names.append(env_name)

        # Stream assignment avoids materializing all env objects at once.
        is_selected = (idx % args.num_env_chunks) == args.env_chunk_id
        if not is_selected:
            env_obj.close()
            continue
        selected_indices.append(idx)

        eval_info, _trajs, renders = evaluate(
            agent=agent,
            env=env_obj,
            action_dim=example_batch["actions"].shape[-1],
            num_eval_episodes=args.num_eval_episodes,
            num_video_episodes=args.num_video_episodes,
            num_parallel_envs=args.num_parallel_envs,
            video_frame_skip=args.video_frame_skip,
        )
        env_obj.close()
        selected_env_results.append(
            {
                "env_idx": idx,
                "env_name": env_name,
                "eval_info": eval_info,
                "renders": renders,
            }
        )

    agg_eval_info = {}
    if selected_env_results:
        metric_keys = [k for k in selected_env_results[0]["eval_info"].keys() if k != "video"]
        for key in metric_keys:
            agg_eval_info[key] = float(np.mean([x["eval_info"][key] for x in selected_env_results]))

    out_dir = os.path.join(run_root, "job_results", f"N_{args.n_val}")
    os.makedirs(out_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"chunk_{args.env_chunk_id:03d}_of_{args.num_env_chunks:03d}_{timestamp}.pkl")
    payload = {
        "metadata": {
            "env_name": args.env_name,
            "task_name": args.task_name,
            "n_val": args.n_val,
            "env_chunk_id": args.env_chunk_id,
            "num_env_chunks": args.num_env_chunks,
            "num_parallel_envs": args.num_parallel_envs,
            "num_eval_episodes": args.num_eval_episodes,
            "num_video_episodes": args.num_video_episodes,
            "video_frame_skip": args.video_frame_skip,
            "wandb_run_name": args.wandb_run_name,
            "wandb_group_name": args.wandb_group_name,
            "wandb_project": args.wandb_project,
            "wandb_entity": args.wandb_entity,
            "wandb_run_id": args.wandb_run_id,
            "classifier_ckpt_num": args.classifier_ckpt_num,
            "classifier_restore_dir": args.classifier_restore_dir,
            "actor_restore_path": args.actor_restore_path,
            "timestamp": timestamp,
        },
        "total_envs": total_envs,
        "all_env_names": all_env_names,
        "selected_env_indices": selected_indices,
        "selected_env_results": selected_env_results,
        "chunk_agg_eval_info": agg_eval_info,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved intermediate job result: {out_path}")
    print(f"N={args.n_val}, chunk={args.env_chunk_id}/{args.num_env_chunks}, envs={len(selected_indices)}")


def parse_args():
    parser = argparse.ArgumentParser(description="BRC single-job evaluator for BERT best-of-N.")
    parser.add_argument("--classifier_type", choices=["language", "success"], required=True)
    parser.add_argument("--classifier_restore_dir", type=str, required=True)
    parser.add_argument("--classifier_ckpt_num", type=int, required=True)
    parser.add_argument("--actor_restore_path", type=str, required=True)
    parser.add_argument("--actor_encoder", type=str, default="combined_encoder_small")

    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--n_val", type=int, required=True)
    parser.add_argument("--env_chunk_id", type=int, required=True)
    parser.add_argument("--num_env_chunks", type=int, required=True)

    parser.add_argument("--wandb_group_name", type=str, required=True)
    parser.add_argument("--wandb_run_name", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, default="multitask_RL")
    parser.add_argument("--wandb_entity", type=str, default="yajatyadav")
    parser.add_argument("--wandb_run_id", type=str, default="")

    parser.add_argument("--horizon_length", type=int, default=5)
    parser.add_argument("--num_eval_episodes", type=int, default=55)
    parser.add_argument("--num_video_episodes", type=int, default=5)
    parser.add_argument("--num_parallel_envs", type=int, default=5)
    parser.add_argument("--video_frame_skip", type=int, default=3)
    parser.add_argument("--language_embedder", type=str, default="bert")
    parser.add_argument(
        "--keys_to_load",
        nargs="+",
        type=str,
        default=["agentview_rgb", "eye_in_hand_rgb", "language", "proprio"],
    )
    parser.add_argument("--actor_seed", type=int, default=0)
    parser.add_argument("--demo_nums_to_use_per_task", nargs="+", type=int, default=[0])

    parser.add_argument(
        "--intermediate_base_dir",
        type=str,
        default="exp/multitask_RL/eval_intermediate/brc_eval_jobs",
        help="Temporary per-job outputs. Final aggregated outputs are written by postprocess script.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_single_job(parse_args())
