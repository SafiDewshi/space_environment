from enum import Enum
from typing import Tuple

import gym
import poliastro
from astropy.units import Quantity
from astropy.time import Time
from gym import spaces
from astropy.coordinates import solar_system_ephemeris
from astropy import time, units as u
from poliastro.bodies import Earth, Sun, Moon, Mercury, Venus, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto, \
    SolarSystemPlanet
from poliastro.ephem import Ephem
import numpy as np
from poliastro.frames import Planes
from poliastro.twobody.orbit import Orbit


class SpaceShipName(Enum):
    DEFAULT = "default"
    LOW_THRUST = "high_thrust"
    HIGH_THRUST = "low_thrust"


class SolarSystem(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 bodies,
                 start_time: Time,
                 time_step,
                 start_body: SolarSystemPlanet,
                 target_body: SolarSystemPlanet,
                 spaceship_name: SpaceShipName = SpaceShipName.DEFAULT,
                 spaceship_initial_altitude: float = 400,
                 spaceship_mass: float = None,
                 spaceship_delta_v: float = None,
                 spaceship_engine_thrust: float = None
                 ):
        super(SolarSystem, self).__init__()

        # set up solar system
        solar_system_ephemeris.set("jpl")
        # Download & use JPL Ephem

        body_dict = {
            "Earth": [Earth, Moon],
            "All":
                [Sun, Earth, Moon, Mercury, Venus, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto]
        }
        # define bodies to model
        # poliastro.bodies.SolarSystemPlanet =
        #   Sun, Earth, Moon, Mercury, Venus, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto
        # could also add versions for: only inner solar system, only 'major' bodies jovan moons, saturn's moons?
        try:
            self.body_list = body_dict[bodies]
        except KeyError:
            raise KeyError(f"bodies must be one of {body_dict.keys()}")

        # system_ephems = self._get_ephem_from_list_of_bodies(body_list)

        # set up spacecraft
        spaceship_initial_altitude = spaceship_initial_altitude * u.km
        self.spaceship = SpaceShip.get_default_ships(spaceship_name, spaceship_initial_altitude, start_time, start_body)
        # override defaults if given
        if spaceship_mass is not None:
            self.spaceship.mass = spaceship_mass
        if spaceship_delta_v is not None:
            self.spaceship.fuel = spaceship_delta_v
        if spaceship_engine_thrust is not None:
            self.spaceship.engine_thrust = spaceship_engine_thrust

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

        # todo: get list of celestial bodies, initial spacecraft position, time?
        # dict: all_planets, earth/moon, inner_planets, w/e
        # observation :
        # time, craft position, craft velocity, craft fuel, craft engine power, bodies: position, velocity, mass
        # [time, [craft position, velocity, fuel, engine power], [target body position, velocity, mass],
        # [other body1 position, vel, mass], [other bodyn position, velocity, mass]]

    def step(self, action):
        observation = []
        reward = 0
        done = 0
        info = []

        # observation should be a list of bodies including their positions and speeds,
        # as well as the spacecraft's position, speed, and fuel?
        # observation :
        # time, craft position, craft velocity, craft fuel, craft engine power, bodies: position, velocity, mass
        # [time, [craft position, velocity, fuel, engine power], [target body position, velocity, mass],
        # [other body1 position, vel, mass], [other bodyn position, velocity, mass]]

        # todo: take input action in the form thrust direction, thrust percentage, thrust duration?

        return observation, reward, done, info

    def reset(self):
        # get planet and spaceship positions at start_time, reset spaceship fuel, 
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
        self.engine_thrust = None
        self.rv = None  # type: Tuple[Quantity, Quantity]

    @classmethod
    def get_default_ships(cls, ship_name: SpaceShipName, altitude, start_time, start_body):
        low_earth_orbit = SpaceShip.from_equatorial_circular_orbit(start_body, altitude, start_time)
        ships = {
            SpaceShipName.DEFAULT:
                SpaceShip(
                    initial_orbit=low_earth_orbit, mass=50, delta_v=50, engine_thrust=50
                ),
            SpaceShipName.HIGH_THRUST:
                SpaceShip(
                    initial_orbit=low_earth_orbit, mass=50, delta_v=50, engine_thrust=100
                ),
            SpaceShipName.LOW_THRUST:
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
