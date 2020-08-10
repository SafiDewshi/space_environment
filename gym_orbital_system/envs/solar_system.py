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
    LOW_THRUST = "low_thrust"
    HIGH_THRUST = "high_thrust"
    # todo: also add a blank ship?


class SystemScope(Enum):
    EARTH = "Earth"
    ALL = "All"


class SolarSystem(gym.Env):
    metadata = {'render.modes': ['human']}

    @u.quantity_input
    def __init__(self,
                 bodies: SystemScope = SystemScope.ALL,
                 start_body: SolarSystemPlanet = None,
                 target_bodies: List[SolarSystemPlanet] = None,
                 start_time: Time = None,
                 time_step: TimeDelta = TimeDelta(1 * u.hour),
                 spaceship_name: SpaceShipName = SpaceShipName.DEFAULT,
                 spaceship_initial_altitude: u.km = 400 * u.km,
                 spaceship_mass: u.kg = None,
                 spaceship_delta_v: u.m / u.s = None,
                 spaceship_engine_thrust: u.N = None
                 ):
        super(SolarSystem, self).__init__()

        if start_body is None:
            start_body = Earth
        if target_bodies is None:
            target_bodies = [Mars]
        if start_time is None:
            start_time = Time(datetime.now()).tdb

        self.start_body = start_body
        self.target_bodies = target_bodies
        self.start_time = start_time
        self.current_time = None
        self.time_step = time_step
        self.done = False
        self.reward = 0
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
        self.spaceship = SpaceShip.get_default_ships(
            spaceship_name, spaceship_initial_altitude, self.start_time, self.start_body
        )
        # override defaults if given
        if spaceship_mass is not None:
            self.spaceship.mass = spaceship_mass
        if spaceship_delta_v is not None:
            self.spaceship.fuel = spaceship_delta_v
        if spaceship_engine_thrust is not None:
            self.spaceship.engine_thrust = spaceship_engine_thrust

        self.current_ephem = None

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

        # [time_step, [craft position, velocity, fuel, engine power],
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
        info = []

        # observation should be a list of bodies including their positions and speeds,
        # as well as the spacecraft's position, speed, and fuel?

        self._record_current_state()

        # todo: take input action in the form thrust direction, thrust time as percentage of time step
        # todo: calculate effect of ship thrust and bodies gravity on ship's rv()
        self.spaceship.calculate_net_force_on_ship()
        self.spaceship.update_ship_rv()

        self.current_time += self.time_step
        # increment time

        self.current_ephem = self._get_ephem_from_list_of_bodies(self.body_list, self.current_time)
        # update system ephem for new time_step

        observation = self._get_observation()
        # return new observation of craft rv, fuel levels, system positions
        # todo: calculate rewards? other info?
        # to calculate rewards - check current position & velocity are within acceptable bounds of target?
        # ^ doesn't work for multiple targets.
        # check rewards over threshold?
        # check all targets have been visited?
        self._calculate_rewards()
        # when target is visited to within desired thresholds, mark it as visited.
        # when all targets are done, set done = True
        self._check_if_done()

        return observation, self.reward, self.done, info

    def reset(self):
        # get planet and spaceship positions at start_time, reset spaceship fuel,

        self.current_time = self.start_time

        self.current_ephem = self._get_ephem_from_list_of_bodies(self.body_list, self.start_time)

        start_body_ephem = Ephem.from_body(self.start_body, self.start_time)
        self.spaceship.global_rv = (
            self.spaceship.local_rv[0] + start_body_ephem.rv()[0],
            self.spaceship.local_rv[1] + start_body_ephem.rv()[1]
        )
        # convert spaceship rv to system-relative rather than earth

        observation = self._get_observation()

        return observation

    def render(self, mode='human'):
        # todo: plot current coordinates of system?
        # todo: maybe plot previous coordinates too?
        pass

    def close(self):
        # todo: ?
        pass

    @staticmethod
    def _get_ephem_from_list_of_bodies(bodies, current_time):
        list_of_bodies = []
        for i in bodies:
            body = Ephem.from_body(i, current_time)
            list_of_bodies.append([i, body])
        return list_of_bodies

    def _get_observation(self):
        obs = [
            self.time_step,
            [self.spaceship.global_rv[0], self.spaceship.global_rv[1], self.spaceship.fuel,
             self.spaceship.engine_thrust]
        ]
        # todo: units!
        # todo: reformat obs into np array
        # [time_step, [craft position, velocity, fuel, engine power],
        # [body_1_is_target, body_1_position, body_1_velocity, body_1_mass],
        # ...
        # [body_n_is_target, body_n_position, body_n_velocity, body_n_mass]]
        for body in self.current_ephem:
            obs.append(
                [body[0] in self.target_bodies, body[0].mass, body[1].rv()[0], body[1].rv()[1]]
            )
        return obs

    def _record_current_state(self):
        # self.current_ephem
        # self.spaceship
        # write relevant info to file?

        # todo: record planet & craft position history
        return

    def _calculate_rewards(self):
        # todo: check if craft fulfils reward criteria - close/slow enough to a target body?
        # if so increment reward and remove is_target boolean
        if False:
            self.reward += 1
        return

    def _check_if_done(self):
        # todo: check if every target has been visited
        if False:
            self.done = True
        return


class SpaceShip:

    def __init__(self, *, initial_orbit, mass, delta_v, engine_thrust):
        self.initial_orbit = initial_orbit
        self.mass = mass
        self.velocity = delta_v
        self.fuel = None
        self.engine_thrust = engine_thrust
        self.local_rv = self.initial_orbit.rv()  # type: Tuple[Quantity, Quantity]
        self.global_rv = (None, None)

    @classmethod
    def get_default_ships(cls, ship_name: SpaceShipName, altitude, start_time, start_body):
        start_orbit = SpaceShip.from_equatorial_circular_orbit(start_body, altitude, start_time)

        # todo: change to quantities
        # todo: add total mass, dry mass, propellant mass, find total mass from that

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

    def calculate_net_force_on_ship(self):
        net_force = []
        # direction and strength of force
        # F = Gm/r^2
        # todo: for each body, calculate F vector (including direction)
        # todo: sum F vectors for each body
        pass

    def update_ship_rv(self):
        # todo: take ship position, velocity, thrust, net_force and update ship position/velocity
        pass
