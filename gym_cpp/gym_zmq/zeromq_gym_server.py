"""
Implement a ZeroMQ-based environment to be able to access from other programming language.
As the Python has GIL, each environment creation from the host language will instantiate a server for parallelism
"""

import zmq
import gym
import os
import tempfile
import argparse
import json
import numpy as np
import six

from .encoding import encode_numpy, decode_numpy


def encode_space(space: gym.spaces.Space):
    if isinstance(space, gym.spaces.Box):
        output = {
            "type": "Box",
            "low": encode_numpy(space.low),
            "high": encode_numpy(space.high),
            "shape": list(space.shape),
            "dtype": space.dtype.name
        }
    elif isinstance(space, gym.spaces.Discrete):
        output = {
            "type": "Discrete",
            "n": space.n
        }
    else:
        raise NotImplementedError(f"Not implemented space encoding for space {space}")
    return json.dumps(output)


def encode_observation(observation):
    if isinstance(observation, np.ndarray):
        return encode_numpy(observation)
    else:
        raise ValueError(f"Unknown observation type {type(observation)}")


def decode_action(action) -> np.ndarray:
    if isinstance(action, six.integer_types):
        nice_action = action
    elif isinstance(action, str):
        nice_action = decode_numpy(action)
    else:
        raise ValueError(f"Unknown encoded action type {type(action)}")
    return nice_action


class ZeroMQGymServer(object):
    def __init__(self, port):
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        directory = os.path.join(tempfile.gettempdir(), 'gym')
        os.makedirs(directory, exist_ok=True)
        self.file = os.path.join(directory, str(port))
        self.url = f"ipc://{self.file}"
        self.socket.bind(self.url)
        self.env = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.socket.unbind(self.url)
        if self.file is not None:
            os.remove(self.file)

    def handle_query(self, query):
        query = json.loads(query)
        command = query["command"]
        if command == "create":
            env_name = query["env_name"]
            self.env = gym.make(env_name)
            return "done"
        elif command == "observation_space":
            return encode_space(self.env.observation_space)
        elif command == "action_space":
            return encode_space(self.env.action_space)
        elif command == "reset":
            obs = self.env.reset()
            return encode_observation(obs)
        elif command == "step":
            action = query["action"]
            action = decode_action(action)
            next_obs, reward, done, info = self.env.step(action)
            output = {
                "next_obs": encode_observation(next_obs),
                "reward": reward,
                "done": done,
                "info": info
            }
            return json.dumps(output)
        elif command == "seed":
            seed = query["seed"]
            self.env.seed(seed)
            return "done"
        else:
            raise ValueError(f"Unknown command {command}")

    def run(self):
        encoding = 'utf-8'
        while True:
            message = self.socket.recv()
            message = message.decode(encoding)
            output = self.handle_query(message)
            output = bytes(output, encoding)
            self.socket.send(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', default=5000, type=int, help='port to bind to')
    args = parser.parse_args()
    with ZeroMQGymServer(port=args.port) as server:
        server.run()
