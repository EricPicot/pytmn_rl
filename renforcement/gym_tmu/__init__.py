from gym.envs.registration import register

register(
    id='tmu-v0',
    entry_point='renforcement.gym_tmu.env:GymTMU',
)
