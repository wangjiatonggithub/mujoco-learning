import numpy as np
import mujoco
import gym
from gym import spaces

# Off-policy algorithm and HER
try:
    from stable_baselines3 import SAC
    from sb3_contrib.her import HerReplayBuffer
except Exception:
    raise ImportError("Please install stable-baselines3 and sb3-contrib: pip install stable-baselines3 sb3-contrib")

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch
import torch.nn as nn
import time
import warnings
from typing import Optional

# Reuse helper functions from original project (flag file handling)
import os

def write_flag_file(flag_filename="rl_visu_flag"):
    flag_path = os.path.join("/tmp", flag_filename)
    try:
        with open(flag_path, "w") as f:
            f.write("This is a flag file")
        return True
    except Exception:
        return False


def check_flag_file(flag_filename="rl_visu_flag"):
    flag_path = os.path.join("/tmp", flag_filename)
    return os.path.exists(flag_path)


def delete_flag_file(flag_filename="rl_visu_flag"):
    flag_path = os.path.join("/tmp", flag_filename)
    if not os.path.exists(flag_path):
        return True
    try:
        os.remove(flag_path)
        return True
    except Exception:
        return False


# ...existing PandaObstacleEnv class should remain in original file; we will import it instead of copying.
# To keep this new file self-contained, we will import the original module if available.

try:
    from rl_panda_obstacle_high_profile import PandaObstacleEnv
except Exception:
    raise ImportError("Could not import PandaObstacleEnv from rl_panda_obstacle_high_profile.py. Make sure that file is in PYTHONPATH and contains PandaObstacleEnv.")


class PandaGoalEnv(gym.GoalEnv):
    """Wrapper that converts PandaObstacleEnv to a GoalEnv for HER.
    observation: dict with keys 'observation', 'achieved_goal', 'desired_goal'
    - 'observation' excludes desired_goal (only joint+obstacles info)
    - 'achieved_goal' is end-effector position (3,)
    - 'desired_goal' is target position (3,)
    """
    def __init__(self, env: PandaObstacleEnv):
        super().__init__()
        self.env = env

        # Build spaces
        # observation (without desired_goal): joint_pos(7) + obstacles positions + sizes
        num_obstacles = self.env.obstacle_positions.shape[0]
        size_dim = self.env.obstacle_sizes.shape[1]
        obs_without_goal_dim = 7 + (3 * num_obstacles) + (size_dim * num_obstacles)

        self.observation_space = spaces.Dict({
            'observation': spaces.Box(low=-np.inf, high=np.inf, shape=(obs_without_goal_dim,), dtype=np.float32),
            'achieved_goal': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            'desired_goal': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        })
        self.action_space = self.env.action_space

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict) -> float:
        dist = np.linalg.norm(achieved_goal - desired_goal)
        return 0.0 if dist < self.env.goal_arrival_threshold else -1.0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.env.reset(seed=seed)
        return self._get_obs_dict()

    def step(self, action):
        # action assumed in [-1,1]^7 as original env
        obs_raw, _, terminated, truncated, info = self.env.step(action)
        achieved = self.env.data.body(self.env.end_effector_id).xpos.copy()
        desired = self.env.goal_position.copy()
        obs_dict = self._get_obs_dict()
        # compute sparse reward consistent with compute_reward
        reward = self.compute_reward(achieved, desired, info)
        done = terminated or truncated or (np.linalg.norm(achieved - desired) < self.env.goal_arrival_threshold)
        return obs_dict, reward, done, False, info

    def _get_obs_dict(self):
        # observation without desired_goal
        joint_pos = self.env.data.qpos[:7].copy().astype(np.float32)
        obstacle_positions = self.env._get_obstacle_centers()
        obstacle_sizes = self.env._get_obstacle_sizes_obs()
        obs_without_goal = np.concatenate([
            joint_pos,
            obstacle_positions.flatten(),
            obstacle_sizes.flatten()
        ]).astype(np.float32)
        achieved = self.env.data.body(self.env.end_effector_id).xpos.copy().astype(np.float32)
        desired = self.env.goal_position.copy().astype(np.float32)
        return {'observation': obs_without_goal, 'achieved_goal': achieved, 'desired_goal': desired}

    def close(self):
        self.env.close()


def train_sac_her(
    n_envs: int = 8,
    total_timesteps: int = 2_000_000,
    model_save_path: str = "panda_sac_her",
    visualize: bool = False,
    obstacle_type: str = "sphere",
    obstacle_randomize_pos: bool = True,
    randomize_init_qpos: bool = False,
    randomize_goal_pos: bool = True,
    max_episode_steps: int = 500,
):
    ENV_KWARGS = {
        'visualize': visualize,
        'obstacle_type': obstacle_type,
        'obstacle_randomize_pos': obstacle_randomize_pos,
        'randomize_init_qpos': randomize_init_qpos,
        'randomize_goal_pos': randomize_goal_pos
    }

    def make_env_fn():
        base = PandaObstacleEnv(**ENV_KWARGS)
        return PandaGoalEnv(base)

    env = make_vec_env(
        env_id=lambda: make_env_fn(),
        n_envs=n_envs,
        seed=42,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "fork"}
    )

    # HER replay buffer kwargs
    replay_buffer_kwargs = dict(
        n_sampled_goal=4, # 每个实际采样轨迹额外多生成多少HER样本
        goal_selection_strategy="future", # 从未来轨迹中采样目标位置
        online_sampling=True, # 在采样时动态生成HER样本而不是预先生成并存储，节省内存但增加采样时间
        max_episode_length=max_episode_steps,
    )

    policy_kwargs = dict(
        activation_fn=nn.LeakyReLU,
        net_arch=[dict(pi=[512, 256, 128], qf=[512, 256, 128])]
    )

    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=1, # 打印训练日志
        buffer_size=1_000_000,
        learning_rate=3e-4,
        batch_size=256,
        tau=0.005, # 目标网络软更新系数
        gamma=0.99,
        train_freq=1, # 每采样1步更新一次策略
        gradient_steps=1, # 每次更新利用1步采样数据
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=replay_buffer_kwargs,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log="./tensorboard/panda_sac_her/"
    )

    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save(model_save_path)
    env.close()
    print(f"模型已保存至: {model_save_path}")


def test_sac_her(
    model_path: str = "panda_sac_her",
    total_episodes: int = 5,
    obstacle_type: str = "sphere",
    obstacle_randomize_pos: bool = True,
    randomize_init_qpos: bool = False,
    randomize_goal_pos: bool = True,
    max_episode_steps: int = 500,
):
    env = PandaGoalEnv(PandaObstacleEnv(
        visualize=True,
        obstacle_type=obstacle_type,
        obstacle_randomize_pos=obstacle_randomize_pos,
        randomize_init_qpos=randomize_init_qpos,
        randomize_goal_pos=randomize_goal_pos,
        max_episode_time=40.0,
        pause_on_collision=True
    ))

    model = SAC.load(model_path, env=env)

    success_count = 0
    for ep in range(total_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0
        while not done and steps < max_episode_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            episode_reward += reward
            steps += 1
        achieved = obs['achieved_goal']
        desired = obs['desired_goal']
        is_success = np.linalg.norm(achieved - desired) < env.env.goal_arrival_threshold
        if is_success:
            success_count += 1
        print(f"Ep {ep+1}: reward={episode_reward:.2f} success={is_success}")

    print(f"Success rate: {success_count}/{total_episodes}")
    env.close()


if __name__ == "__main__":
    TRAIN_MODE = True
    OBSTACLE_TYPE = "box"
    OBSTACLE_RANDOMIZE_POS = True
    RANDOMIZE_INIT_QPOS = True
    RANDOMIZE_GOAL_POS = True

    delete_flag_file()
    MODEL_PATH = "assets/model/rl_obstacle_avoidance_checkpoint/panda_sac_her_v1"

    if TRAIN_MODE:
        train_sac_her(
            n_envs=8,
            total_timesteps=2_000_000,
            model_save_path=MODEL_PATH,
            visualize=False,
            obstacle_type=OBSTACLE_TYPE,
            obstacle_randomize_pos=OBSTACLE_RANDOMIZE_POS,
            randomize_init_qpos=RANDOMIZE_INIT_QPOS,
            randomize_goal_pos=RANDOMIZE_GOAL_POS,
            max_episode_steps=500,
        )
    else:
        test_sac_her(
            model_path=MODEL_PATH,
            total_episodes=20,
            obstacle_type=OBSTACLE_TYPE,
            obstacle_randomize_pos=OBSTACLE_RANDOMIZE_POS,
            randomize_init_qpos=RANDOMIZE_INIT_QPOS,
            randomize_goal_pos=RANDOMIZE_GOAL_POS,
            max_episode_steps=500,
        )
