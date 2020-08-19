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
from astropy.constants import G, g0
from poliastro.bodies import Earth, Sun, Moon, Mercury, Venus, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto, \
    SolarSystemPlanet
from poliastro.ephem import Ephem
import numpy as np
from poliastro.frames import Planes
from poliastro.twobody.orbit import Orbit


class SpaceShipName(Enum):
    DEFAULT = "default, approximately Cassini-Huygens with increased dV (4600) to compensate for lack of 3rd stage"
    LOW_THRUST = "low_thrust, approximately Deep Space 1 with ~6500 m/s dV"
    HIGH_THRUST = "high_thrust"
    # todo: also add a blank ship?


class SystemScope(Enum):
    EARTH = "Earth"
    ALL = "All"
    # add more systems?


class SolarSystem(gym.Env):
    metadata = {'render.modes': ['human']}

    @u.quantity_input
    def __init__(self,
                 bodies: SystemScope = SystemScope.ALL,
                 start_body: SolarSystemPlanet = None,
                 target_bodies: List[SolarSystemPlanet] = None,
                 start_time: Time = None,
                 action_step: TimeDelta = TimeDelta(1 * u.hour),
                 simulation_step: TimeDelta = TimeDelta(1 * u.minute),
                 spaceship_name: SpaceShipName = SpaceShipName.DEFAULT,
                 spaceship_initial_altitude: u.km = 400 * u.km,
                 spaceship_mass: u.kg = None,
                 spaceship_propellant_mass: u.kg = None,
                 spaceship_isp: u.s = None,
                 spaceship_engine_thrust: u.N = None
                 ):
        super(SolarSystem, self).__init__()

        if start_body is None:
            start_body = Earth
        if target_bodies is None:
            target_bodies = [Mars]
        if start_time is None:
            start_time = Time(datetime.now()).tdb

        # todo: enforce action_step/simulation_step is an integer?

        self.start_body = start_body
        self.target_bodies = target_bodies
        self.spaceship_initial_altitude = spaceship_initial_altitude
        self.start_time = start_time
        self.current_time = None
        self.time_step = action_step
        self.simulation_step = simulation_step
        self.done = False
        self.reward = 0
        self.done = False

        self.spaceship_name = spaceship_name
        self.spaceship_mass = spaceship_mass
        self.spaceship_propellant_mass = spaceship_propellant_mass
        self.spaceship_isp = spaceship_isp
        self.spaceship_engine_thrust = spaceship_engine_thrust

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

        self.spaceship = self._init_spaceship()

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

        # todo: take input action in the form thrust direction, thrust time as percentage of time step
        # todo: calculate effect of ship thrust and bodies gravity on ship's rv()

        self._update_ship_dynamics(action)

        observation = self._get_observation()

        self._record_current_state()

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

        # set up spacecraft
        self.spaceship = self._init_spaceship()

        start_body_ephem = Ephem.from_body(self.start_body, self.start_time)
        self.spaceship.rv = (
            self.spaceship.rv[0] + start_body_ephem.rv()[0],
            self.spaceship.rv[1] + start_body_ephem.rv()[1]
        )
        # convert spaceship rv to system-relative rather than earth

        observation = self._get_observation()

        return observation

    def _init_spaceship(self) -> 'SpaceShip':
        spaceship = SpaceShip.get_default_ships(
            self.spaceship_name, self.spaceship_initial_altitude, self.start_time, self.start_body
        )
        # override defaults if given
        if self.spaceship_mass is not None:
            spaceship.mass = self.spaceship_mass
        if self.spaceship_propellant_mass is not None:
            spaceship.propellant_mass = self.spaceship_propellant_mass
        if self.spaceship_isp is not None:
            spaceship.isp = self.spaceship_isp
        if self.spaceship_engine_thrust is not None:
            spaceship.thrust = self.spaceship_engine_thrust
        return spaceship

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
            [self.spaceship.rv[0], self.spaceship.rv[1], self.spaceship.propellant_mass,
             self.spaceship.thrust]
        ]
        # todo: rework with units!
        # todo: reformat obs into np array, need better shape
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

        # todo: record planet & craft position history for later use/plotting
        return

    def _calculate_rewards(self):
        # todo: check if craft fulfils reward criteria - close/slow enough to a target body?
        # if so increment reward and remove is_target boolean
        # support for flybys? assign rewards for any flybys? careful if the system learns to just flyby one body?
        # negative score for each time step to encourage reaching end.

        if True:
            self.reward += 1
        return

    def _check_if_done(self):
        # todo: check if every target has been visited
        # if done, adjust reward based off elapsed time and remaining fuel
        if True:
            self.done = True
        return

    def _update_ship_dynamics(self, action):

        # todo: update ship position in smaller time steps than self.timestep
        #  since an orbit might take 90m but self.timestep defaults to 60m

        n_iter = int(np.ceil((self.time_step / self.simulation_step).value))
        ship_position, ship_velocity = self.spaceship.rv
        for x in range(0, n_iter):
            grav_acc = self._calculate_gravitational_acceleration()
            dv, direction = self._calculate_engine_delta_v(action)
            dv_vector = dv * direction
            ship_position = self._update_position(ship_position, ship_velocity, grav_acc, dv_vector)
            ship_velocity = self._update_velocity(ship_velocity, grav_acc, dv_vector)
            self.current_time += self.simulation_step
            # increment time
            self.current_ephem = self._get_ephem_from_list_of_bodies(self.body_list, self.current_time)
            # update system ephem for new time_step
            # todo: get next 30min of ephem at once

        pass

    def _update_position(self, position, velocity, grav_acc, delta_v):
        return position + (velocity + 0.5 * grav_acc * self.simulation_step + 0.5 * delta_v) * self.simulation_step

    def _update_velocity(self, velocity, grav_acc, delta_v):
        return velocity + delta_v + grav_acc * self.simulation_step

    def _calculate_engine_delta_v(self, action):
        direction, thrust_percent = action
        direction = direction / np.linalg.norm(direction)
        if thrust_percent != 0:
            thrust_time = self.simulation_step / thrust_percent
        else:
            thrust_time = 0.
        exhaust_velocity = self.spaceship.isp * g0
        mass_flow_rate = self.spaceship.thrust / exhaust_velocity
        delta_m = mass_flow_rate * thrust_time
        mass_start = self.spaceship.total_mass
        mass_final = mass_start - delta_m
        delta_v = exhaust_velocity * np.log((mass_start / mass_final))
        self.spaceship.total_mass = mass_final

        return delta_v, direction

    def _calculate_gravitational_acceleration(self):
        # direction and strength of force
        # F = Gm/r^2
        acceleration = []
        for body in self.current_ephem:
            r_vector = body[1].rv()[0] - self.spaceship.rv[0]
            r_magnitude = np.linalg.norm(r_vector)

            a_magnitude = (G.to("km3/kg s2") * body[0].mass) / (r_magnitude ** 2)
            a_vector = a_magnitude * r_vector / r_magnitude
            acceleration.append(a_vector)

        total_acceleration = 0
        for a in acceleration:
            total_acceleration += a

        return total_acceleration


class SpaceShip:

    def __init__(self, *, initial_orbit, dry_mass, propellant_mass, isp, thrust):
        self.initial_orbit = initial_orbit
        self.dry_mass = dry_mass
        self.propellant_mass = propellant_mass
        self.isp = isp
        self.thrust = thrust
        self.rv = self.initial_orbit.rv()  # type: Tuple[Quantity, Quantity]

    @property
    def total_mass(self):
        return self.dry_mass + self.propellant_mass

    @total_mass.setter
    def total_mass(self, total_mass):
        self.propellant_mass = total_mass - self.dry_mass

    @classmethod
    def get_default_ships(cls, ship_name: SpaceShipName, altitude, start_time, start_body):
        start_orbit = SpaceShip.from_equatorial_circular_orbit(start_body, altitude, start_time)

        ships = {
            SpaceShipName.DEFAULT:
                SpaceShip(
                    initial_orbit=start_orbit,
                    dry_mass=2500 * u.kg,
                    propellant_mass=10000 * u.kg,
                    # propellant mass is significantly increased to compensate for third stage
                    isp=300 * u.s,
                    thrust=500 * u.N
                    # approx dV = 4600
                ),
            SpaceShipName.HIGH_THRUST:
                SpaceShip(
                    initial_orbit=start_orbit,
                    dry_mass=2500 * u.kg,
                    propellant_mass=10000 * u.kg,
                    isp=300 * u.s,
                    thrust=5000 * u.N
                ),
            SpaceShipName.LOW_THRUST:
                SpaceShip(
                    initial_orbit=start_orbit,
                    dry_mass=400 * u.kg,
                    propellant_mass=100 * u.kg,
                    isp=3000 * u.s,
                    thrust=0.1 * u.N
                    # approx dV = 6500
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
