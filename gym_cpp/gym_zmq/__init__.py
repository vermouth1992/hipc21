from gym import envs
from gym.envs.registration import register

# register all the environments that are already registered by appending name "ZeroMQ"
all_envs = list(envs.registry.all())

for env in all_envs:
    attributes = env.__dict__.copy()
    attributes["kwargs"] = {'env_name': attributes["id"]}
    attributes["id"] = 'ZMQ' + attributes["id"]
    attributes["entry_point"] = 'gym_zmq.zeromq_gym_client:ZeroMQEnv'
    to_be_deleted = ['_env_name', '_kwargs']
    for a in to_be_deleted:
        if a in attributes:
            attributes.pop(a)
    register(
        **attributes
    )
