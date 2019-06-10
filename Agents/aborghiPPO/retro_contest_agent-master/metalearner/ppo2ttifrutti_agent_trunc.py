#!/usr/bin/env python

"""
Train an agent on Sonic using PPO2ttifrutti, a variant of OpenAI PPO2 baseline.
"""

import tensorflow as tf
import numpy as np
import gym
import gym_remote.exceptions as gre
import os
import math

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

import ppo2ttifrutti
import ppo2ttifrutti_policies as policies
import ppo2ttifrutti_sonic_env as env

def main():
    """Run PPO until the environment throws an exception."""
    config = tf.ConfigProto()
    #os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    config.gpu_options.allow_growth = True # pylint: disable=E1101

    with tf.Session(config=config):
        # Take more timesteps than we need to be sure that
        # we stop due to an exception.
        ppo2ttifrutti.learn(policy=policies.CnnPolicy,
                            env=SubprocVecEnv([env.make_train_5, env.make_train_18, env.make_train_43]),
                            nsteps=2048,
                            nminibatches=16,
                            lam=0.95,
                            gamma=0.99,
                            noptepochs=4,
                            log_interval=1,
                            ent_coef=0.01,
                            lr=lambda _: 2e-4,
                            cliprange=lambda _: 0.1,
                            total_timesteps=int(1e9),
                            save_interval=25)

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
