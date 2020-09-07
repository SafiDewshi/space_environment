from gym.envs.registration import register

register(
    id='solarsystem-v0',
    entry_point='gym_orbital_system.envs:SolarSystem',
)
