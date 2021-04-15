from gym.envs.registration import register

register(
    id='fault-v0',
    entry_point='gym_fault.envs:FaultEnv',
    max_episode_steps=1000,
)