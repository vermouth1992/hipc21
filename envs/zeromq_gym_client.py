#
#   Hello World client in Python
#   Connects REQ socket to tcp://localhost:5555
#   Sends "Hello" to server, expects "World" back
#

import gym
import numpy as np
import zmq
import subprocess
import tempfile
import os
import signal
import json
import io
import base64
import six


def encode_numpy(array: np.ndarray) -> str:
    b = io.BytesIO()
    np.savez(b, x=array)
    encoded = base64.b64encode(b.getvalue()).decode('ascii')
    return encoded


def decode_numpy(string: str) -> np.ndarray:
    decoded = base64.b64decode(string)
    buffer = io.BytesIO(decoded)
    return np.load(buffer)['x']


def decode_space(space):
    space = json.loads(space)
    type = space["type"]
    if type == "Box":
        low = decode_numpy(space["low"])
        high = decode_numpy(space["high"])
        shape = space["shape"]
        dtype = space["dtype"]
        return gym.spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
    elif type == "Discrete":
        n = space["n"]
        return gym.spaces.Discrete(n=n)
    else:
        raise NotImplementedError(f"Unsupported space type {type}")


def encode_action(action):
    if isinstance(action, np.ndarray):
        action = encode_numpy(action)
    elif isinstance(action, six.integer_types):
        pass
    else:
        raise ValueError(f"Unknown action type {type(action)}")
    return action


class ZeroMQEnv(gym.Env):
    def __init__(self, env_name):
        # create a server and client
        self.directory = os.path.join(tempfile.gettempdir(), 'gym')
        self.port = self.find_available_port()
        self.server_process = self.create_server(port=self.port)
        self.client_socket = self.create_client_socket()
        # send request to the server to make an environment
        # query the observation space and action space
        message = self._query({"command": "create", "env_name": env_name})
        assert message == "done"
        obs_space_str = self._query({"command": "observation_space"})
        self.observation_space = decode_space(obs_space_str)
        act_space_str = self._query({"command": "action_space"})
        self.action_space = decode_space(act_space_str)

    def find_available_port(self):
        max_trial = 100
        for _ in range(max_trial):
            i = np.random.randint(10000)
            path = os.path.join(self.directory, str(i))
            if not os.path.exists(path):
                return i
        raise ValueError(f"Can't find available port in {max_trial} trials")

    def create_server(self, port):
        server_file = "zeromq_gym_server.py"
        command = "python {} -p {}".format(server_file, port)
        server_process = subprocess.Popen(command.split(), shell=False)
        return server_process

    def create_client_socket(self):
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(f"ipc://{self.directory}/{self.port}")
        return socket

    def __del__(self):
        os.kill(self.server_process.pid, signal.SIGTERM)

    def _query(self, query):
        encoding = 'utf-8'
        query = json.dumps(query)
        query = bytes(query, encoding)
        self.client_socket.send(query)
        message = self.client_socket.recv()
        return message.decode(encoding)

    def reset(self):
        obs = self._query({"command": "reset"})
        return decode_numpy(obs)

    def step(self, action):
        message = self._query({"command": "step", "action": encode_action(action)})
        # decode message
        message = json.loads(message)
        next_obs = decode_numpy(message["next_obs"])
        reward = message["reward"]
        done = message["done"]
        info = message["info"]
        return next_obs, reward, done, info

    def seed(self, seed=None):
        message = self._query({"command": "seed", "seed": seed})
        assert message == "done"


def run_env(env_fn):
    env = env_fn()
    env.seed(10)
    env.action_space.seed(10)
    done = False
    obs = env.reset()
    total_rewards = 0.
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_rewards += reward
    print(f"Total rewards: {total_rewards}")


if __name__ == '__main__':
    env_name = 'Pong-v4'
    zeromq_env_fn = lambda: ZeroMQEnv(env_name=env_name)
    env_fn = lambda: gym.make(env_name)
    import timeit

    result = timeit.timeit(lambda: run_env(zeromq_env_fn), number=10)
    print(result)
    result = timeit.timeit(lambda: run_env(env_fn), number=10)
    print(result)
