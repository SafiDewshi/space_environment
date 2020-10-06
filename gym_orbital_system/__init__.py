from gym.envs.registration import register

register(
    id='solarsystem-v0',
    entry_point='gym_orbital_system.envs:SolarSystem',
),
register(
    id='earth-v0',
    entry_point='gym_orbital_system.envs:EarthSystem',
),
register(
    id='solarsystemgrav-v0',
    entry_point='gym_orbital_system.envs:SolarSystemGrav',
),
register(
    id='solarsystemfull-v0',
    entry_point='gym_orbital_system.envs:SolarSystemFull',
)
