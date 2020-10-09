from datetime import datetime
from enum import Enum

import gym
from astropy.time import Time
from gym import spaces
from astropy.coordinates import solar_system_ephemeris
from astropy import units as u
from astropy.constants import g0
from poliastro.bodies import Earth
from poliastro.maneuver import Maneuver
import numpy as np
from poliastro.twobody.orbit import Orbit


class SpaceShipName(Enum):
    CASSINI = "default, approximately Cassini-Huygens with increased dV (4600) to compensate for lack of 3rd stage"
    LOW_THRUST = "low_thrust, approximately Deep Space 1 with ~6500 m/s dV"
    TEST_SHIP = "Test ship with high thrust fuel and isp"


class SystemScope(Enum):
    EARTH = "simple orbit around Earth with a target orbit"
    # add more systems?


class DummyBody:
    def __init__(self, name):
        self.name = name


class EarthSystem(gym.Env):
    metadata = {'render.modes': ['human']}

    @u.quantity_input
    def __init__(self,
                 bodies: SystemScope = SystemScope.EARTH,
                 start_time: Time = None,
                 action_step: u.s = 60 * u.s,
                 simulation_ratio: int = 1,
                 number_of_steps: int = 100000,
                 spaceship_name: SpaceShipName = SpaceShipName.TEST_SHIP,
                 spaceship_initial_altitude: u.km = 400 * u.km,
                 spaceship_target_altitude: u.km = 35_786 * u.km,  # geostationary
                 spaceship_mass: u.kg = None,
                 spaceship_propellant_mass: u.kg = None,
                 spaceship_isp: u.s = None,
                 spaceship_engine_thrust: u.N = None
                 ):
        super(EarthSystem, self).__init__()

        self.last_step_ship_proximity = {}
        if start_time is None:
            start_time = Time(datetime.now()).tdb

        # enforce action_step/simulation_step is an integer
        if action_step.value % simulation_ratio != 0:
            raise ValueError("Action step must be evenly divisible by simulation_ratio")

        self.spaceship_initial_altitude = spaceship_initial_altitude
        self.spaceship_target_altitude = spaceship_target_altitude
        self.start_time = start_time
        self.current_time = start_time
        self.action_step = action_step
        self.simulation_ratio = simulation_ratio
        self.number_of_steps = number_of_steps
        self.done = False
        self.reward = 0
        self.initial_reset = False
        self.ephems = {}

        self.spaceship_name = spaceship_name
        self.spaceship_mass = spaceship_mass
        self.spaceship_propellant_mass = spaceship_propellant_mass
        self.spaceship_isp = spaceship_isp
        self.spaceship_engine_thrust = spaceship_engine_thrust

        # set up solar system

        body_dict = {
            SystemScope.EARTH: {
                "attractor": Earth,
                "SPK": "jpl"
            }
        }

        try:
            self.body_dict = body_dict[bodies]
        except KeyError:
            raise KeyError(f"bodies must be one of {body_dict.keys()}")

        solar_system_ephemeris.set(self.body_dict["SPK"])
        # Download & set Ephem

        # set up spacecraft

        self.spaceship = self._init_spaceship()

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(8,))

        # action:
        # [x,y,z, burn duration]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,))  # x,y,z direction vector, burn duration as
        # percent of time_step

        self.r_normalization = 2e6 * u.km
        self.v_normalization = 25 * u.km / u.s  # factor to divide velocity observations by to normalize them to +- 1.
        # 2xEarth's escape velocity

        self.finish_time = self.start_time + self.action_step * self.number_of_steps

    @property
    def simulation_step(self):
        return self.action_step / self.simulation_ratio

    def step(self, action):

        info = []
        self.reward = 0
        self.done = False

        if not self.initial_reset:
            return self.reset(), self.reward, self.done, info

        dv = self._calculate_action_delta_v(action)
        # take action in the form of a direction and burn time fraction, output total delta v
        maneuvers = self._split_burn_to_impulse_list(dv)
        # split the delta v into impulses for each simulation step to smooth the impulse. returns list of impulses
        self._apply_maneuvers(maneuvers)
        # apply the impulses for maneuver, including propagating the orbit forwards to the end of that
        self.current_time = self.spaceship.orbit.epoch

        observation = self._get_observation()

        # self._record_current_state()

        self._assign_all_rewards()

        return observation, self.reward, self.done, info

    def reset(self):
        self.done = False
        # get planet and spaceship positions at start_time, reset spaceship fuel,
        self.current_time = self.start_time
        self.reward = 0

        # set up spacecraft
        self.spaceship = self._init_spaceship()

        observation = self._get_observation()

        self.initial_reset = True

        return observation

    def _init_spaceship(self) -> 'SpaceShip':
        spaceship = SpaceShip.get_default_ships(
            self.spaceship_name, self.spaceship_initial_altitude, self.start_time, self.body_dict["attractor"]
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
        pass

    def _get_observation(self):

        ship_r = self.spaceship.orbit.r.decompose().value / self.r_normalization.decompose().value
        ship_v = self.spaceship.orbit.v.decompose().value / self.v_normalization.decompose().value
        obs = [
            self.spaceship.total_mass.decompose().value / self.spaceship.initial_mass.decompose().value,
            self.spaceship.propellant_mass.decompose().value / self.spaceship.initial_mass.decompose().value,
            *ship_r,
            *ship_v,
        ]

        np_obs = np.array(obs)
        return np_obs

    # def _record_current_state(self):
    #     # self.current_ephem
    #     # self.spaceship
    #     # write timestamp, orbits, and action
    #
    #     return

    def _calculate_action_delta_v(self, action):
        *direction, thrust_percent = action
        thrust_percent = (thrust_percent / 2) + 0.5
        direction = direction / np.linalg.norm(direction)
        thrust_time = self.simulation_step * thrust_percent
        exhaust_velocity = self.spaceship.isp * g0
        mass_flow_rate = self.spaceship.thrust / exhaust_velocity
        delta_m = (mass_flow_rate * thrust_time).decompose()
        mass_start = self.spaceship.total_mass
        mass_final = mass_start - delta_m
        if mass_final < self.spaceship.dry_mass:
            mass_final = self.spaceship.dry_mass
        delta_v = exhaust_velocity * np.log((mass_start / mass_final))
        self.spaceship.total_mass = mass_final

        return delta_v * direction

    def _split_burn_to_impulse_list(self, dv):
        split_impulse = dv / self.simulation_ratio
        # applying these maneuvers takes a long time; reduce simulation ratio?
        impulse = []
        for x in range(0, self.simulation_ratio):
            impulse.append((self.simulation_step, split_impulse))
        maneuvers = Maneuver(*impulse)
        return maneuvers

    def _apply_maneuvers(self, maneuvers):
        self.spaceship.previous_orbit = self.spaceship.orbit
        self.spaceship.orbit = self.spaceship.orbit.apply_maneuver(maneuvers).propagate(self.simulation_step)


    def _assign_all_rewards(self):
        self._calculate_proximity_rewards()
        self._check_remaining_rewards()

    def _calculate_proximity_rewards(self):
        apo_current = self.spaceship.orbit.r_a
        apo_prev = self.spaceship.previous_orbit.r_a
        peri_current = self.spaceship.orbit.r_p
        peri_prev = self.spaceship.previous_orbit.r_p
        target = self.spaceship_target_altitude
        current_apo_diff = abs(target - apo_current)
        prev_apo_diff = abs(target - apo_prev)
        current_peri_diff = abs(target - peri_current)
        prev_peri_diff = abs(target - peri_prev)
        if current_apo_diff < prev_apo_diff:
            self.reward += 5
        # reward proportional to the amount the orbit's furthest point has approached target
        if current_peri_diff < prev_peri_diff:
            self.reward += 5
        # reward proportional to the amount the orbit's nearest point has approached target

    def _check_remaining_rewards(self):
        remaining_fuel_fraction = self.spaceship.propellant_mass / self.spaceship.initial_propellant
        if self.spaceship.orbit.a < self.body_dict["attractor"].R:
            self.done = True
            self.reward -= 100
        elif self.current_time > self.finish_time:
            self.done = True
            self.reward -= 10
        elif remaining_fuel_fraction < 0:
            self.done = True
            self.reward -= 10
        elif abs(self.spaceship_target_altitude - self.spaceship.orbit.a) < 10 * u.km \
                and self.spaceship.orbit.ecc < 0.05:
            self.done = True
            self.reward += 10000*remaining_fuel_fraction


class SpaceShip:

    def __init__(self, *, orbit, dry_mass, propellant_mass, isp, thrust):
        self.orbit = orbit
        self.previous_orbit = orbit
        self.dry_mass = dry_mass
        self.propellant_mass = propellant_mass
        self.isp = isp
        self.thrust = thrust
        self.initial_mass = dry_mass + propellant_mass
        self.initial_propellant = propellant_mass

    @property
    def total_mass(self):
        return self.dry_mass + self.propellant_mass

    @total_mass.setter
    def total_mass(self, total_mass):
        self.propellant_mass = total_mass - self.dry_mass

    @classmethod
    def get_default_ships(cls, ship_name: SpaceShipName, altitude, start_time, start_body):
        start_orbit = Orbit.circular(start_body, alt=altitude, epoch=start_time)

        ships = {
            SpaceShipName.CASSINI:
                SpaceShip(
                    orbit=start_orbit,
                    dry_mass=2500 * u.kg,
                    propellant_mass=10000 * u.kg,
                    # propellant mass is significantly increased to compensate for third stage
                    isp=300 * u.s,
                    thrust=500 * u.N
                    # approx dV = 4600
                ),
            SpaceShipName.TEST_SHIP:
                SpaceShip(
                    orbit=start_orbit,
                    dry_mass=2500 * u.kg,
                    propellant_mass=10000 * u.kg,
                    isp=300000 * u.s,
                    thrust=500 * u.N
                ),
            SpaceShipName.LOW_THRUST:
                SpaceShip(
                    orbit=start_orbit,
                    dry_mass=400 * u.kg,
                    propellant_mass=100 * u.kg,
                    isp=3000 * u.s,
                    thrust=0.1 * u.N
                    # approx dV = 6500
                )
        }

        return ships.get(ship_name)

    def propagate(self, *args, **kwargs):
        return self.orbit.propagate(*args, **kwargs)
