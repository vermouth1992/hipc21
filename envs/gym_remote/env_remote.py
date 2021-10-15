"""
Remote environments with client defined here. This is to make fair comparison with C++ based implementation
"""

import os
import signal
import subprocess
import time

import gym
import numpy as np
import torch

from .gym_http_client import Client
from .gym_http_server import encode_tensor_base64, decode_tensor

_PORT = 5000


def get_available_port():
    global _PORT
    port = _PORT
    _PORT += 1
    return port


def get_space_from_info(space_info):
    type = space_info.get('name', None)
    if type == 'Box':
        space = gym.spaces.Box(low=decode_tensor(space_info['low']).numpy(),
                               high=decode_tensor(space_info['high']).numpy())
    elif type == 'Discrete':
        space = gym.spaces.Discrete(n=space_info['n'])
    else:
        raise ValueError("Unknown space {}".format(type))

    return space


class RemoteEnv(gym.Env):
    def __init__(self, env_id):
        """
        For each environment instance, we need to create a separate server in order to avoid conflict
        """
        port = get_available_port()
        remote_base = 'http://127.0.0.1:{}'.format(port)
        self.client = Client(remote_base)
        # use a subprocess to start a server listening on the port
        # get the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        server_file = os.path.join(current_dir, "gym_http_server.py")
        command = "python {} -p {}".format(server_file, port)

        self.server_process = subprocess.Popen(command.split(), shell=False, )
        # stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        time.sleep(1.5)
        self.instance_id = self.client.env_create(env_id)

        # query observation_space and action_space
        self.observation_space = get_space_from_info(self.client.env_observation_space_info(self.instance_id))
        self.action_space = get_space_from_info(self.client.env_action_space_info(self.instance_id))

    def __del__(self):
        # self.client.env_close(self.instance_id)
        # self.server_process.terminate()
        os.kill(self.server_process.pid, signal.SIGTERM)

    def step(self, action):
        # from IPython import embed
        # embed()

        if isinstance(action, np.ndarray):
            action = torch.as_tensor(action)
            action = encode_tensor_base64(action)
        elif isinstance(action, np.int64):
            action = action.item()
        elif isinstance(action, int):
            pass
        else:
            raise ValueError("Unknown action type", type(action))

        observation, reward, done, info = self.client.env_step(self.instance_id, action, False)
        observation = decode_tensor(observation).numpy()
        return observation, reward, done, info

    def reset(self):
        obs = self.client.env_reset(self.instance_id)
        return decode_tensor(obs).numpy()
