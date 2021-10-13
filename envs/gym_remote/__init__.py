from gym import envs
from gym.envs.registration import register

# register all the environments that are already registered by appending name "Remote"
all_envs = list(envs.registry.all())

for env in all_envs:
    attributes = env.__dict__.copy()
    attributes["kwargs"] = {'env_id': attributes["id"]}
    attributes["id"] = 'Remote' + attributes["id"]
    attributes["entry_point"] = 'gym_remote.env_remote:RemoteEnv'
    to_be_deleted = ['_env_name', '_kwargs']
    for a in to_be_deleted:
        if a in attributes:
            attributes.pop(a)
    register(
        **attributes
    )
