from datetime import datetime
from enum import Enum
from typing import Tuple, List

import gym
import poliastro
from astropy.units import Quantity
from astropy.time import Time, TimeDelta
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
    # todo: also add a blank ship?


class SystemScope(Enum):
    EARTH = "Earth"
    ALL = "All"


class SolarSystem(gym.Env):
    metadata = {'render.modes': ['human']}

    @u.quantity_input
    def __init__(self,
                 bodies: SystemScope,
                 start_time: Time,
                 start_body: SolarSystemPlanet,
                 target_bodies: List[SolarSystemPlanet],
                 time_step: TimeDelta = TimeDelta(1 * u.hour),
                 spaceship_name: SpaceShipName = SpaceShipName.DEFAULT,
                 spaceship_initial_altitude: u.km = 400 * u.km,
                 spaceship_mass: u.kg = None,
                 spaceship_delta_v: u.m/u.s = None,
                 spaceship_engine_thrust: u.N = None
                 ):
        super(SolarSystem, self).__init__()

        if not target_bodies:
            raise ValueError("Target bodies must be a list of one or more poliastro.bodies.SolarSystemPlanet")

        self.start_time = start_time
        self.time_step = time_step
        self.done = False


        # set up solar system
        solar_system_ephemeris.set("jpl")
        # Download & use JPL Ephem

        body_dict = {
            SystemScope.EARTH: [Earth, Moon],
            SystemScope.ALL:
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

        # set up spacecraft
        spaceship_initial_altitude = spaceship_initial_altitude
        self.spaceship = SpaceShip.get_default_ships(spaceship_name, spaceship_initial_altitude, start_time, start_body)
        # override defaults if given
        if spaceship_mass is not None:
            self.spaceship.mass = spaceship_mass
        if spaceship_delta_v is not None:
            self.spaceship.fuel = spaceship_delta_v
        if spaceship_engine_thrust is not None:
            self.spaceship.engine_thrust = spaceship_engine_thrust

        self.start_ephem = None

        # init:
        # * which bodies are modelled
        # * what time it is
        # * what time_step to use
        # * target body
        # * spaceship pos/vel (orbit?) /fuel/thrust
        # *

        # init must define action & observation space
        # initialize model solar system
        #
        # Define action and observation space
        # They must be gym.spaces objects

        # observation ~~time~~, time_step, craft position, craft velocity, craft fuel, craft engine power,
        # bodies: position, velocity, mass

        # [time_step [craft position, velocity, fuel, engine power],
        # [body_1_is_target, body_1_position, body_1_velocity, body_1_mass],
        # ...
        # [body_n_is_target, body_n_position, body_n_velocity, body_n_mass]]
        self.observation_space = spaces.Space()

        # action:
        # tuple [[x,y,z], burn duration]
        self.action_space = spaces.Tuple((
            spaces.Box(low=-1.0, high=1.0, shape=(3,)),  # x,y,z direction vector
            spaces.Box(low=0.0, high=1.0, shape=(1,))  # burn duration as percent of time_step
        ))

    def step(self, action):
        observation = []
        reward = 0
        done = 0
        info = []

        # observation should be a list of bodies including their positions and speeds,
        # as well as the spacecraft's position, speed, and fuel?

        # todo: take input action in the form thrust direction, thrust percentage, thrust duration?
        # todo: calculate effect of ship thrust and bodies gravity on ship's rv()
        # todo: update system ephem for new time_step
        # todo: return new observation of craft rv, fuel levels, system positions. Write log of ship & system positions?
        # todo: calculate rewards? other info?
        # todo: record position history

        return observation, reward, done, info

    def reset(self):
        # get planet and spaceship positions at start_time, reset spaceship fuel,

        self.start_ephem = self._get_ephem_from_list_of_bodies(self.body_list, self.start_time)

        # system_ephem = self._get_ephem_from_list_of_bodies(body_list, time)

        observation = []

        return observation

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    # todo: get list of planets and their positions using

    @staticmethod
    def _get_ephem_from_list_of_bodies(bodies, current_time):
        list_of_bodies = []
        for i in bodies:
            body = Ephem.from_body(i, current_time)
            list_of_bodies.append([i.name, body])
        return list_of_bodies


class SpaceShip:

    def __init__(self, *, initial_orbit, mass, delta_v, engine_thrust):
        self.initial_orbit = initial_orbit
        self.mass = mass
        self.velocity = delta_v
        self.fuel = None
        self.engine_thrust = engine_thrust
        self.rv = self.initial_orbit.rv()  # type: Tuple[Quantity, Quantity]

    @classmethod
    def get_default_ships(cls, ship_name: SpaceShipName, altitude, start_time, start_body):
        start_orbit = SpaceShip.from_equatorial_circular_orbit(start_body, altitude, start_time)
        ships = {
            SpaceShipName.DEFAULT:
                SpaceShip(
                    initial_orbit=start_orbit, mass=50, delta_v=50, engine_thrust=50
                ),
            SpaceShipName.HIGH_THRUST:
                SpaceShip(
                    initial_orbit=start_orbit, mass=50, delta_v=50, engine_thrust=100
                ),
            SpaceShipName.LOW_THRUST:
                SpaceShip(
                    initial_orbit=start_orbit, mass=50, delta_v=50, engine_thrust=5
                )
        }

        return ships.get(ship_name)

    @classmethod
    def from_start_orbit(cls, body, altitude, eccentricity, inclination, raan, argp, nu, epoch):
        return Orbit.from_classical(body, altitude, eccentricity, inclination, raan, argp, nu, epoch)

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
            epoch=start_time
        )
