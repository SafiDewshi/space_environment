from typing import Tuple

import gym
import poliastro
from astropy.units import Quantity
from gym import spaces
from astropy.coordinates import solar_system_ephemeris
from astropy import time, units as u
from poliastro.bodies import Earth, Sun, Moon, Mercury, Venus, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto
from poliastro.ephem import Ephem
import numpy as np
from poliastro.frames import Planes
from poliastro.twobody.orbit import Orbit


class SolarSystem(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, bodies, time, timestep, target_body, spaceship):
        super(SolarSystem, self).__init__()

        # init:
        # * which bodies are modelled
        # * what time it is
        # * what timestep to use
        # * target body
        # * spaceship pos/vel (orbit?) /fuel/thrust
        # *

        # init must define action & observation space
        # initialize model solar system
        #
        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Space()
        self.observation_space = spaces.Space()
        if solar_system_ephemeris.kernel.origin != 'jpl':
            print("downloading high precision jpl ephemerides")
            solar_system_ephemeris.set("jpl")

        # todo: get list of celestial bodies, initial spacecraft position, time?
        # dict: all_planets, earth/moon, inner_planets, w/e
        # poliastro.bodies.SolarSystemPlanet =
        #   Sun, Earth, Moon, Mercury, Venus, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto
        body_dict = {
            "Earth": [Earth, Moon],
            "All":
                [Sun, Earth, Moon, Mercury, Venus, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto]
        }

        # observation :
        # time, craft position, craft velocity, craft fuel, craft engine power, bodies: position, velocity, mass
        # [time, [craft position, velocity, fuel, engine power], [target body position, velocity, mass],
        # [other body1 position, vel, mass], [other bodyn position, velocity, mass]]

        try:
            body_list = body_dict[bodies]
        except KeyError:
            print(f"bodies must be one of {body_dict.keys()}")
            return

        system_ephems = self._get_ephem_from_list_of_bodies(body_list)

    def step(self, action):
        observation = []
        reward = 0
        done = 0
        info = []

        # observation should be a list of bodies including their positions and speeds,
        # as well as the spacecraft's position, speed, and fuel?

        # todo: take input action in the form thrust direction, thrust percentage, thrust duration?

        return observation, reward, done, info

    def reset(self):
        observation = []

        return observation

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    # todo: get list of planets and their positions using

    def _get_ephem_from_list_of_bodies(self, bodies):
        list_of_bodies = []
        for i in bodies:
            body = Ephem.from_body(i)
            list_of_bodies.append(body)
        return list_of_bodies


class SpaceShip:
    def __init__(self, *, initial_orbit, mass, delta_v, engine_thrust):
        self.initial_orbit = None
        self.mass = None
        self.velocity = None
        self.fuel = None
        self.engine_thrust_max = None
        self.engine_thrust_min = None
        self.rv = None  # type: Tuple[Quantity, Quantity]

    @classmethod
    def get_default_ships(cls, ship_name, altitude, start_time):
        low_earth_orbit = SpaceShip.from_equatorial_circular_orbit(Earth, altitude * u.km, start_time)
        ships = {
            "default_ship":
                SpaceShip(
                    initial_orbit=low_earth_orbit, mass=50, delta_v=50, engine_thrust=50
                ),
            "high_thrust":
                SpaceShip(
                    initial_orbit=low_earth_orbit, mass=50, delta_v=50, engine_thrust=100
                ),
            "low_thrust":
                SpaceShip(
                    initial_orbit=low_earth_orbit, mass=50, delta_v=50, engine_thrust=5
                )
        }

        return ships.get(ship_name)

    @classmethod
    def from_start_orbit(cls, body, altitude, eccentricity, inclination, raan, argp, nu, epoch, plane):
        return Orbit.from_classical(body, altitude, eccentricity, inclination, raan, argp, nu, epoch, plane)

    @classmethod
    def from_equatorial_circular_orbit(cls, body, altitude, start_time):
        return cls.from_start_orbit(
            body,
            altitude,
            eccentricity=0 * u.one,
            inclination=0 * u.deg,
            raan=0 * u.deg,
            argp=0 * u.deg,
            nu=0 * u.deg,
            epoch=start_time)
