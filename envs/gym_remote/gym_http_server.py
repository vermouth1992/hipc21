import flask
from flask import Flask, request, jsonify

assert flask.__version__ == '2.0.1'

import uuid
import gym
import numpy as np
import six
import argparse
import base64
import json
import torch
import io

import logging

logger = logging.getLogger('werkzeug')
logger.setLevel(logging.ERROR)


def encode_tensor_base64(tensor: torch.Tensor) -> str:
    b = io.BytesIO()
    torch.save(tensor, b)
    encoded = base64.b64encode(b.getvalue()).decode('ascii')
    return encoded


def decode_tensor(string: str) -> torch.Tensor:
    decoded = base64.b64decode(string)
    buffer = io.BytesIO(decoded)
    return torch.load(buffer)


########## Container for environments ##########
class Envs(object):
    """
    Container and manager for the environments instantiated
    on this server.

    When a new environment is created, such as with
    envs.create('CartPole-v0'), it is stored under a short
    identifier (such as '3c657dbc'). Future API calls make
    use of this instance_id to identify which environment
    should be manipulated.
    """

    def __init__(self):
        self.envs = {}
        self.id_len = 8

    def _lookup_env(self, instance_id):
        try:
            return self.envs[instance_id]
        except KeyError:
            raise InvalidUsage('Instance_id {} unknown'.format(instance_id))

    def _remove_env(self, instance_id):
        try:
            del self.envs[instance_id]
        except KeyError:
            raise InvalidUsage('Instance_id {} unknown'.format(instance_id))

    def create(self, env_id, seed=None):
        try:
            env = gym.make(env_id)
            if seed:
                env.seed(seed)
        except gym.error.Error:
            raise InvalidUsage("Attempted to look up malformed environment ID '{}'".format(env_id))

        instance_id = str(uuid.uuid4().hex)[:self.id_len]
        self.envs[instance_id] = env
        print(self.list_all())
        return instance_id

    def list_all(self):
        return dict([(instance_id, env.spec.id) for (instance_id, env) in self.envs.items()])

    def reset(self, instance_id):
        env = self._lookup_env(instance_id)
        obs = env.reset()
        obs_tensor = torch.from_numpy(obs)
        return encode_tensor_base64(obs_tensor)

    def step(self, instance_id, action, render):
        env = self._lookup_env(instance_id)
        if isinstance(action, six.integer_types):
            nice_action = action
        else:
            # decode it
            nice_action = decode_tensor(action).numpy()
        if render:
            env.render()
        [observation, reward, done, info] = env.step(nice_action)
        return [encode_tensor_base64(torch.from_numpy(observation)), reward, done, info]

    def get_action_space_contains(self, instance_id, x):
        env = self._lookup_env(instance_id)
        return env.action_space.contains(int(x))

    def get_action_space_info(self, instance_id):
        env = self._lookup_env(instance_id)
        return self._get_space_properties(env.action_space)

    def get_action_space_sample(self, instance_id):
        env = self._lookup_env(instance_id)
        action = env.action_space.sample()
        if isinstance(action, (list, tuple)) or ('numpy' in str(type(action))):
            try:
                action = action.tolist()
            except TypeError:
                print(type(action))
                print('TypeError')
        return action

    def get_observation_space_contains(self, instance_id, j):
        env = self._lookup_env(instance_id)
        info = self._get_space_properties(env.observation_space)
        for key, value in j.items():
            # Convert both values to json for comparibility
            if json.dumps(info[key]) != json.dumps(value):
                print('Values for "{}" do not match. Passed "{}", Observed "{}".'.format(key, value, info[key]))
                return False
        return True

    def get_observation_space_info(self, instance_id):
        env = self._lookup_env(instance_id)
        return self._get_space_properties(env.observation_space)

    def _get_space_properties(self, space):
        info = {}
        info['name'] = space.__class__.__name__
        if info['name'] == 'Discrete':
            info['n'] = space.n
        elif info['name'] == 'Box':
            info['shape'] = space.shape
            # It's not JSON compliant to have Infinity, -Infinity, NaN.
            # Many newer JSON parsers allow it, but many don't. Notably python json
            # module can read and write such floats. So we only here fix "export version",
            # also make it flat.
            info['low'] = [(float(x) if x != -np.inf else -1e100) for x in np.array(space.low).flatten()]
            info['low'] = encode_tensor_base64(torch.tensor(info['low']))
            info['high'] = [(float(x) if x != +np.inf else +1e100) for x in np.array(space.high).flatten()]
            info['high'] = encode_tensor_base64(torch.tensor(info['high']))
        elif info['name'] == 'HighLow':
            info['num_rows'] = space.num_rows
            info['matrix'] = [((float(x) if x != -np.inf else -1e100) if x != +np.inf else +1e100) for x in
                              np.array(space.matrix).flatten()]
        return info

    def env_close(self, instance_id):
        env = self._lookup_env(instance_id)
        env.close()
        self._remove_env(instance_id)
        print(self.list_all())


########## App setup ##########
app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
envs = Envs()


########## Error handling ##########
class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv


def get_required_param(json, param):
    if json is None:
        logger.info("Request is not a valid json")
        raise InvalidUsage("Request is not a valid json")
    value = json.get(param, None)
    if (value is None) or (value == '') or (value == []):
        logger.info("A required request parameter '{}' had value {}".format(param, value))
        raise InvalidUsage("A required request parameter '{}' was not provided".format(param))
    return value


def get_optional_param(json, param, default):
    if json is None:
        logger.info("Request is not a valid json")
        raise InvalidUsage("Request is not a valid json")
    value = json.get(param, None)
    if (value is None) or (value == '') or (value == []):
        logger.info(
            "An optional request parameter '{}' had value {} and was replaced with default value {}".format(param,
                                                                                                            value,
                                                                                                            default))
        value = default
    return value


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


########## API route definitions ##########
@app.route('/v1/envs/', methods=['POST'])
def env_create():
    """
    Create an instance of the specified environment

    Parameters:
        - env_id: gym environment ID string, such as 'CartPole-v0'
        - seed: set the seed for this env's random number generator(s).
    Returns:
        - instance_id: a short identifier (such as '3c657dbc')
        for the created environment instance. The instance_id is
        used in future API calls to identify the environment to be
        manipulated
    """
    env_id = get_required_param(request.get_json(), 'env_id')
    seed = get_optional_param(request.get_json(), 'seed', None)
    instance_id = envs.create(env_id, seed)
    return jsonify(instance_id=instance_id)


@app.route('/v1/envs/', methods=['GET'])
def env_list_all():
    """
    List all environments running on the server

    Returns:
        - envs: dict mapping instance_id to env_id
        (e.g. {'3c657dbc': 'CartPole-v0'}) for every env
        on the server
    """
    all_envs = envs.list_all()
    return jsonify(all_envs=all_envs)


@app.route('/v1/envs/<instance_id>/reset/', methods=['POST'])
def env_reset(instance_id):
    """
    Reset the state of the environment and return an initial
    observation.

    Parameters:
        - instance_id: a short identifier (such as '3c657dbc')
        for the environment instance
    Returns:
        - observation: the initial observation of the space
    """
    observation = envs.reset(instance_id)
    return jsonify(observation=observation)


@app.route('/v1/envs/<instance_id>/step/', methods=['POST'])
def env_step(instance_id):
    """
    Run one timestep of the environment's dynamics.

    Parameters:
        - instance_id: a short identifier (such as '3c657dbc')
        for the environment instance
        - action: an action to take in the environment
    Returns:
        - observation: agent's observation of the current
        environment
        - reward: amount of reward returned after previous action
        - done: whether the episode has ended
        - info: a dict containing auxiliary diagnostic information
    """
    json = request.get_json()
    action = get_required_param(json, 'action')
    render = get_optional_param(json, 'render', False)
    [obs_jsonable, reward, done, info] = envs.step(instance_id, action, render)
    timeout = info.get('TimeLimit.truncated', False)
    return jsonify(observation=obs_jsonable,
                   reward=reward, done=done, info=info, timeout=timeout)


@app.route('/v1/envs/<instance_id>/action_space/', methods=['GET'])
def env_action_space_info(instance_id):
    """
    Get information (name and dimensions/bounds) of the env's
    action_space

    Parameters:
        - instance_id: a short identifier (such as '3c657dbc')
        for the environment instance
    Returns:
    - info: a dict containing 'name' (such as 'Discrete'), and
    additional dimensional info (such as 'n') which varies from
    space to space
    """
    info = envs.get_action_space_info(instance_id)
    return jsonify(info=info)


@app.route('/v1/envs/<instance_id>/action_space/sample', methods=['GET'])
def env_action_space_sample(instance_id):
    """
    Get a sample from the env's action_space

    Parameters:
        - instance_id: a short identifier (such as '3c657dbc')
        for the environment instance
    Returns:

    	- action: a randomly sampled element belonging to the action_space
    """
    action = envs.get_action_space_sample(instance_id)
    return jsonify(action=action)


@app.route('/v1/envs/<instance_id>/action_space/contains/<x>', methods=['GET'])
def env_action_space_contains(instance_id, x):
    """
    Assess that value is a member of the env's action_space
    
    Parameters:
        - instance_id: a short identifier (such as '3c657dbc')
        for the environment instance
	    - x: the value to be checked as member
    Returns:
        - member: whether the value passed as parameter belongs to the action_space
    """

    member = envs.get_action_space_contains(instance_id, x)
    return jsonify(member=member)


@app.route('/v1/envs/<instance_id>/observation_space/contains', methods=['POST'])
def env_observation_space_contains(instance_id):
    """
    Assess that the parameters are members of the env's observation_space

    Parameters:
        - instance_id: a short identifier (such as '3c657dbc')
        for the environment instance
    Returns:
        - member: whether all the values passed belong to the observation_space
    """
    j = request.get_json()
    member = envs.get_observation_space_contains(instance_id, j)
    return jsonify(member=member)


@app.route('/v1/envs/<instance_id>/observation_space/', methods=['GET'])
def env_observation_space_info(instance_id):
    """
    Get information (name and dimensions/bounds) of the env's
    observation_space

    Parameters:
        - instance_id: a short identifier (such as '3c657dbc')
        for the environment instance
    Returns:
        - info: a dict containing 'name' (such as 'Discrete'),
        and additional dimensional info (such as 'n') which
        varies from space to space
    """
    info = envs.get_observation_space_info(instance_id)
    return jsonify(info=info)


@app.route('/v1/envs/<instance_id>/close/', methods=['POST'])
def env_close(instance_id):
    """
    Manually close an environment

    Parameters:
        - instance_id: a short identifier (such as '3c657dbc')
          for the environment instance
    """
    envs.env_close(instance_id)
    response = f"Successfully close instance {instance_id}"
    return jsonify(info=response)


@app.route('/v1/shutdown/', methods=['POST'])
def shutdown():
    """ Request a server shutdown - currently used by the integration tests to repeatedly create and destroy fresh copies of the server running in a separate thread"""
    f = request.environ.get('werkzeug.server.shutdown')
    f()
    return jsonify(info='Server shutting down')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start a Gym HTTP API server')
    parser.add_argument('-l', '--listen', help='interface to listen to', default='127.0.0.1')
    parser.add_argument('-p', '--port', default=5000, type=int, help='port to bind to')

    args = parser.parse_args()
    print('Server starting at: ' + 'http://{}:{}'.format(args.listen, args.port))
    app.run(host=args.listen, port=args.port)