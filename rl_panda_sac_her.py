import numpy as np
import mujoco
# try:
import gymnasium as gym
from gymnasium import spaces
# except Exception:
# import gym
# from gym import spaces

# Off-policy algorithm and HER
# try:
from stable_baselines3 import SAC
from stable_baselines3 import HerReplayBuffer
# from sb3_contrib.her import HerReplayBuffer
# except Exception:
#     raise ImportError("Please install stable-baselines3 and sb3-contrib: pip install stable-baselines3 sb3-contrib")

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

# try:
from rl_panda_obstacle_high_profile import PandaObstacleEnv
# except Exception:
#     raise ImportError("Could not import PandaObstacleEnv from rl_panda_obstacle_high_profile.py. Make sure that file is in PYTHONPATH and contains PandaObstacleEnv.")


GoalEnvBase = getattr(gym, "GoalEnv", gym.Env)


class PandaGoalEnv(GoalEnvBase):
    """Wrapper that converts PandaObstacleEnv to a GoalEnv for HER.
    observation: dict with keys 'observation', 'achieved_goal', 'desired_goal'
    - 'observation' excludes desired_goal (only joint+obstacles info)
    - 'achieved_goal' is end-effector position (3,)
    - 'desired_goal' is target position (3,)
    """
    def __init__(self, env: PandaObstacleEnv):
        super().__init__()
        self.env = env

        # shaping and penalties (defaults)
        self.gamma = 0.99  # discount used for potential-based shaping
        self.shaping_scale = 1.0  # scale for potential-based shaping term
        self.collision_penalty = -10.0  # additional penalty when collision occurs

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

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info) -> np.ndarray:
        """Return reward composed of:
        - sparse success reward (0 success / -1 failure)
        - collision penalty (if info['collision'] True)
        - (commented) potential-based shaping term: shaping_scale * (prev_dist - gamma * dist)
        Supports both single-step and vectorized inputs (list of info dicts).
        """
        achieved_goal = np.asarray(achieved_goal)
        desired_goal = np.asarray(desired_goal)

        # distances (vectorized)
        dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1)

        # info handling
        if isinstance(info, (list, tuple)):
            prev_dist = np.array([i.get('prev_distance', d) if isinstance(i, dict) else d for i, d in zip(info, np.atleast_1d(dist))], dtype=np.float32)
            collision = np.array([bool(i.get('collision', False)) if isinstance(i, dict) else False for i in info], dtype=bool)
        elif isinstance(info, dict):
            prev_dist = np.array(info.get('prev_distance', dist), dtype=np.float32)
            collision = np.array(info.get('collision', False), dtype=bool)
        else:
            prev_dist = np.array(dist, dtype=np.float32)
            collision = np.zeros_like(dist, dtype=bool)

        # sparse success (success only if within threshold and no collision)
        success = (~collision) & (dist < self.env.goal_arrival_threshold)
        sparse_reward = np.where(success, 0.0, -1.0)

        # collision penalty
        coll_pen = np.where(collision, self.collision_penalty, 0.0)

        # potential-based shaping (phi = -distance)
        # shaping = self.shaping_scale * (prev_dist - self.gamma * dist)

        # total = sparse_reward + coll_pen + shaping
        total = sparse_reward + coll_pen
        return float(total) if np.isscalar(total) else total.astype(np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        result = self.env.reset(seed=seed, options=options)
        if isinstance(result, tuple) and len(result) == 2:
            _, info = result
        else:
            info = {}
        return self._get_obs_dict(), info

    def step(self, action):
        # record previous achieved and distance so that replay buffer / HER can access it
        prev_achieved = self.env.data.body(self.env.end_effector_id).xpos.copy()
        desired = self.env.goal_position.copy()
        prev_dist = float(np.linalg.norm(prev_achieved - desired))

        # compute minimum distance to obstacles at previous step and store for filtering
        try:
            prev_min_dist = float(self.env._min_distance_to_obstacles(prev_achieved))
        except Exception:
            prev_min_dist = float(np.inf)

        # action assumed in [-1,1]^7 as original env
        obs_raw, _, terminated, truncated, info = self.env.step(action)

        # ensure info contains previous distance and min-dist for replay buffer
        info = dict(info) if info is not None else {}
        info['prev_achieved'] = prev_achieved
        info['prev_distance'] = prev_dist
        info['min_dist_to_obstacle'] = prev_min_dist

        achieved = self.env.data.body(self.env.end_effector_id).xpos.copy()
        obs_dict = self._get_obs_dict()

        # compute reward using compute_reward (includes sparse + collision + shaping)
        reward = self.compute_reward(achieved, desired, info)

        # done if underlying env terminated/truncated, collision occurred, or success
        collision = bool(info.get('collision', False))
        dist = float(np.linalg.norm(achieved - desired))
        success = (not collision) and (dist < self.env.goal_arrival_threshold)
        done = bool(terminated or truncated or collision or success)

        return obs_dict, float(reward), done, False, info

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


class FilteredHerReplayBuffer(HerReplayBuffer):
    """HER replay buffer that filters out relabeled samples which were collected
    after a collision or whose stored min distance to obstacles is below a threshold.
    This class wraps HerReplayBuffer.sample and attempts resampling up to a limit
    to produce a full batch of valid samples.
    """
    def __init__(self, *args, min_dist_threshold: float = 0.02, max_filter_attempts: int = 5, max_goal_resample_attempts: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_dist_threshold = float(min_dist_threshold)
        self.max_filter_attempts = int(max_filter_attempts) # 最多采样的batch次数
        # When a relabeled desired_goal is unsafe, how many times to try resampling a different goal
        self.max_goal_resample_attempts = int(max_goal_resample_attempts) # 选定一个transition数据后，最多尝试选取desired_goal的次数

    def sample(self, batch_size: int, env=None):
        """HER采样, 并确保采样的transition数据和desired_goal都不靠近障碍物。"""
        """Oversample-and-select scheme with additional safety check for relabeled goals.
        - repeatedly call parent sample() up to `max_filter_attempts` times
        - collect only transitions whose own info shows no collision and min_dist >= threshold
        - additionally, when relabeled desired_goal is present in the returned batch,
          ensure that that goal point is also sufficiently far from obstacles
        - stop when we have `batch_size` valid transitions and return them
        - fallback to last parent's batch if not enough valid transitions collected
        """
        collected = {}  # dict[key] -> list of values (or dict of lists for nested dicts)
        collected_count = 0
        attempts = 0

        def _batch_to_dict(b):
            if isinstance(b, dict):
                return b
            if hasattr(b, "_asdict"):
                return b._asdict()
            if hasattr(b, "__dict__"):
                return dict(vars(b))
            return {}

        def _normalize_val(v):
            if isinstance(v, torch.Tensor):
                return v
            return np.asarray(v)

        def _stack_list(vals):
            if len(vals) == 0:
                return vals
            if isinstance(vals[0], torch.Tensor):
                return torch.stack(vals, dim=0)
            return np.stack(vals, axis=0)

        def _to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x)

        def _assign_goal(container, key, idx, cand):
            if not isinstance(container, dict):
                return
            if key not in container:
                return
            target = container[key]
            if isinstance(target, torch.Tensor):
                cand_t = torch.as_tensor(cand, device=target.device, dtype=target.dtype)
                target[idx] = cand_t
            else:
                target[idx] = cand

        def _append(collected, key, val): # 为字典collected添加键key和值val
            # Normalize and append a single element `val` under `key` in collected
            if isinstance(val, dict):
                d = collected.setdefault(key, {})
                for subk, subv in val.items():
                    d.setdefault(subk, []).append(_normalize_val(subv))
            else:
                collected.setdefault(key, []).append(_normalize_val(val))

        last_batch = None
        last_batch_raw = None
        while collected_count < batch_size and attempts < self.max_filter_attempts:
            batch_raw = super().sample(batch_size, env=env)  # 父类采样函数，在sb3中定义
            batch = _batch_to_dict(batch_raw)
            last_batch = batch
            last_batch_raw = batch_raw
            infos = batch.get('infos', None)
            if infos is None:
                return batch_raw

            # Try to find relabeled desired goals in the batch (various sb3 layouts)
            desired_goals = None
            # Common sb3 keys: 'observations' (dict) or 'obs'
            obs = batch.get('observations', None)
            if isinstance(obs, dict):
                desired_goals = obs.get('desired_goal', None)
            if desired_goals is None:
                obs2 = batch.get('obs', None)
                if isinstance(obs2, dict):
                    desired_goals = obs2.get('desired_goal', None)
            # Sometimes 'next_observations' may carry goals depending on implementation
            if desired_goals is None:
                next_obs = batch.get('next_observations', None)
                if isinstance(next_obs, dict):
                    desired_goals = next_obs.get('desired_goal', None)

            # helper to compute min distance for a 3D point using env (with wrapper fallbacks)
            def _min_dist_point(pt):
                if pt is None:
                    return -np.inf
                try:
                    # ensure numpy array and shape (3,) if possible
                    pt_arr = np.asarray(pt)
                    if pt_arr.size == 0:
                        return -np.inf
                    # try common wrapper attributes to reach underlying env
                    target_envs = [env, getattr(env, 'env', None), getattr(env, 'unwrapped', None)]
                    for candidate in target_envs:
                        if candidate is None:
                            continue
                        if hasattr(candidate, '_min_distance_to_obstacles'):
                            try:
                                return float(candidate._min_distance_to_obstacles(pt_arr))
                            except Exception:
                                pass
                except Exception:
                    pass
                return -np.inf

            # build mask considering transition info; for unsafe relabeled goals we'll try to resample targets
            mask = []
            for idx, info in enumerate(infos): # enumerate函数同时提取索引和值
                if info is None:
                    mask.append(True)
                    continue
                collision = bool(info.get('collision', False))
                min_dist = float(info.get('min_dist_to_obstacle', -np.inf))

                # If transition itself is bad, reject immediately
                if collision or (min_dist < self.min_dist_threshold):
                    mask.append(False)
                    continue

                # transition is OK; now ensure relabeled desired_goal (if present) is safe.
                goal_safe = True
                if desired_goals is not None:
                    try:
                        dg = _to_numpy(desired_goals[idx])
                        dg_min_dist = _min_dist_point(dg)
                        if dg_min_dist >= self.min_dist_threshold:
                            goal_safe = True
                        else:
                            # try to resample a safe desired_goal without discarding this transition
                            goal_safe = False
                            for try_i in range(self.max_goal_resample_attempts):
                                single_raw = super().sample(1, env=env)
                                single = _batch_to_dict(single_raw)
                                # extract candidate goal from single
                                cand = None
                                s_obs = single.get('observations', None)
                                if isinstance(s_obs, dict):
                                    cand = s_obs.get('desired_goal', None)
                                if cand is None:
                                    s_obs2 = single.get('obs', None)
                                    if isinstance(s_obs2, dict):
                                        cand = s_obs2.get('desired_goal', None)
                                if cand is None:
                                    s_next = single.get('next_observations', None)
                                    if isinstance(s_next, dict):
                                        cand = s_next.get('desired_goal', None)
                                if cand is None:
                                    continue
                                cand = _to_numpy(cand).squeeze()
                                cand_min = _min_dist_point(cand)
                                if cand_min >= self.min_dist_threshold:
                                    # accept this candidate and write back into batch so reward can be recomputed
                                    goal_safe = True
                                    try:
                                        _assign_goal(batch.get('observations', None), 'desired_goal', idx, cand)
                                        _assign_goal(batch.get('next_observations', None), 'desired_goal', idx, cand)
                                        _assign_goal(batch.get('obs', None), 'desired_goal', idx, cand)
                                    except Exception:
                                        pass
                                    break
                    except Exception:
                        goal_safe = True

                mask.append(bool(goal_safe))

            # collect passing indices
            for idx, ok in enumerate(mask):
                if not ok:
                    continue
                for k, v in batch.items():
                    if isinstance(v, dict):
                        _append(collected, k, {subk: subv[idx] for subk, subv in v.items()})
                    else:
                        _append(collected, k, v[idx])
                collected_count += 1
                if collected_count >= batch_size:
                    break

            attempts += 1

        # If we collected enough valid samples, build final batch dict and return
        if collected_count >= batch_size:
            final = {}
            for k, val_list in collected.items():
                if isinstance(val_list, dict):
                    final[k] = {subk: _stack_list(sublist) for subk, sublist in val_list.items()}
                else:
                    final[k] = _stack_list(val_list)
            if isinstance(last_batch_raw, dict) or last_batch_raw is None:
                return final
            try:
                return type(last_batch_raw)(**final)
            except Exception:
                return last_batch_raw

        # Fallback
        return last_batch_raw if last_batch_raw is not None else last_batch

def train_sac_her(
    n_envs: int = 8,
    total_timesteps: int = 2_000_000,
    model_save_path: str = "panda_sac_her",
    visualize: bool = True,
    obstacle_type: str = "sphere",
    obstacle_randomize_pos: bool = True,
    randomize_init_qpos: bool = False,
    randomize_goal_pos: bool = True,
    max_episode_steps: int = 500,
    min_dist_threshold: float = 0.02,  # 新增：用于 FilteredHerReplayBuffer 的距离阈值（米）
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
        min_dist_threshold=min_dist_threshold,  # 传递到 FilteredHerReplayBuffer，用于过滤靠近障碍物的 relabeled 样本
    )

    policy_kwargs = dict(
        activation_fn=nn.LeakyReLU,
        net_arch=dict(pi=[512, 256, 128], qf=[512, 256, 128])
    )

    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=1, # 打印训练日志
        buffer_size=1_000_000,
        learning_starts=2*max_episode_steps,  # 至少跑完一个episode后再开始采样/训练
        learning_rate=3e-4,
        batch_size=256,
        tau=0.005, # 目标网络软更新系数
        gamma=0.99,
        train_freq=1, # 每采样1步更新一次策略
        gradient_steps=1, # 每次更新利用1步采样数据
        replay_buffer_class=FilteredHerReplayBuffer,
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
    min_dist_threshold: float = 0.02,
    render_sleep: float = 0.1,
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

    # Ensure HER replay buffer class/kwargs match the training setup
    custom_objects = {
        "replay_buffer_class": FilteredHerReplayBuffer,
        "replay_buffer_kwargs": dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
            min_dist_threshold=min_dist_threshold,
        ),
    }
    model = SAC.load(model_path, env=env, custom_objects=custom_objects)

    success_count = 0
    for ep in range(total_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0
        last_info = {}
        while not done and steps < max_episode_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            if render_sleep > 0:
                time.sleep(render_sleep)
            last_info = info
            episode_reward += reward
            steps += 1
        achieved = obs['achieved_goal']
        desired = obs['desired_goal']
        # success only if within threshold AND no collision occurred in the episode
        is_success = (np.linalg.norm(achieved - desired) < env.env.goal_arrival_threshold) and (not last_info.get('collision', False))
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
            visualize=True,
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
