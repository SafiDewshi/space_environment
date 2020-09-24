from datetime import datetime
from enum import Enum
from typing import Tuple, List, Dict

import gym
from astropy.units import Quantity
from astropy.time import Time
from gym import spaces
from astropy.coordinates import solar_system_ephemeris, CartesianRepresentation, ICRS
from astropy import units as u
from astropy.constants import g0, G
from poliastro.bodies import Earth, Sun, Moon, Mercury, Venus, Mars, Jupiter, Saturn, Uranus, Neptune, \
    SolarSystemPlanet, Body
from poliastro.ephem import Ephem
from poliastro.maneuver import Maneuver
import numpy as np
from poliastro.twobody.orbit import Orbit
from poliastro.threebody.soi import laplace_radius
from poliastro.util import time_range, norm
from astropy.coordinates import get_body_barycentric_posvel

moon_dict = {
    601: {"body": Body(Saturn, (G * 4e19 * u.kg), "Mimas", R=396 * u.km), "orbital_radius": 185_537 * u.km,
          "soi": None},
    602: {"body": Body(Saturn, (G * 1.1e20 * u.kg), "Enceladus", R=504 * u.km), "orbital_radius": 237_948 * u.km,
          "soi": None},
    603: {"body": Body(Saturn, (G * 6.2e20 * u.kg), "Tethys", R=1_062 * u.km), "orbital_radius": 294_619 * u.km,
          "soi": None},
    604: {"body": Body(Saturn, (G * 1.1e21 * u.kg), "Dione", R=1_123 * u.km), "orbital_radius": 377_396 * u.km,
          "soi": None},
    605: {"body": Body(Saturn, (G * 2.3e21 * u.kg), "Rhea", R=1_527 * u.km), "orbital_radius": 527_108 * u.km,
          "soi": None},
    606: {"body": Body(Saturn, (G * 1.35e23 * u.kg), "Titan", R=5_149 * u.km), "orbital_radius": 1_221_870 * u.km,
          "soi": None},
    608: {"body": Body(Saturn, (G * 1.8e21 * u.kg), "Iapetus", R=1_470 * u.km), "orbital_radius": 3_560_820 * u.km,
          "soi": None}
}


class SpaceShipName(Enum):
    CASSINI = "default, approximately Cassini-Huygens with increased dV (4600) to compensate for lack of 3rd stage"
    LOW_THRUST = "low_thrust, approximately Deep Space 1 with ~6500 m/s dV"
    TEST_SHIP = "Test ship with high thrust fuel and isp"


class SystemScope(Enum):
    PLANETS = "Sun and Planets"
    SATURN = "Saturnian system"
    # add more systems?


class SolarSystem(gym.Env):
    metadata = {'render.modes': ['human']}

    @u.quantity_input
    def __init__(self,
                 bodies: SystemScope = SystemScope.SATURN,
                 start_time: Time = None,
                 action_step: u.s = 3600 * u.s,
                 simulation_ratio: int = 60,
                 number_of_steps: int = 1000,
                 spaceship_name: SpaceShipName = SpaceShipName.LOW_THRUST,
                 spaceship_initial_altitude: u.km = 400 * u.km,
                 spaceship_mass: u.kg = None,
                 spaceship_propellant_mass: u.kg = None,
                 spaceship_isp: u.s = None,
                 spaceship_engine_thrust: u.N = None
                 ):
        super(SolarSystem, self).__init__()

        if start_time is None:
            start_time = Time(datetime.now()).tdb

        # enforce action_step/simulation_step is an integer
        if action_step.value % simulation_ratio != 0:
            raise ValueError("Action step must be evenly divisible by simulation_ratio")

        self.spaceship_initial_altitude = spaceship_initial_altitude
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
            # SystemScope.EARTH: {
            #     "attractor": Earth,
            #     "bodies": [Moon],
            #     "SPK": "jpl"
            # },
            # SystemScope.PLANETS: {
            #     "attractor": Sun,
            #     "bodies": [Earth, Mercury, Venus, Mars, Jupiter, Saturn, Uranus, Neptune],
            #     "SPK": "jpl"
            # },
            SystemScope.SATURN: {
                "attractor": Saturn,
                "bodies": [606, 605, 608, 604, 603, 602, 601],  # Titan, Rhea, Iapetus, Dione, Tethys, Enceladus, Mimas
                "SPK": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/sat427.bsp"
            }
        }

        try:
            self.body_dict = body_dict[bodies]
        except KeyError:
            raise KeyError(f"bodies must be one of {body_dict.keys()}")

        solar_system_ephemeris.set(self.body_dict["SPK"])
        # Download & set Ephem

        # if self.body_dict["attractor"].name == "Sun":
        #     if start_body is None:
        #         start_body = Earth
        #     if target_bodies is None:
        #         target_bodies = [Mars]
        #     self.start_body = start_body
        #     self.target_bodies = []
        #     for target in target_bodies:
        #         self.target_bodies.append(target.name)
        #     self.soi_radii = self._calculate_system_laplace_radii()
        #     self.current_soi = self.start_body.name
        #     self.ephems = self._get_ephem_from_list_of_bodies(self.body_dict, epochs)
        #     self.visited_times = {self.start_body.name: Time(datetime.now()).tdb}

        self._calculate_laplace_radii()
        self.current_soi = self.body_dict["attractor"].name
        self.visited_times = {}

        # set up spacecraft

        self.spaceship = self._init_spaceship()

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(len(self.body_dict["bodies"]) + 2, 9))

        # action:
        # tuple [[x,y,z], burn duration]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,))  # x,y,z direction vector, burn duration as
        # percent of time_step

        self.r_normalization = 54.5e6 * u.km  # factor to divide position observations by to normalize them to +- 1.
        # slightly more than neptune's orbit in km?
        self.v_normalization = 100 * u.km / u.s  # factor to divide velocity observations by to normalize them to +- 1.
        # approx 2x mercury's orbital velocity in km?

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

        # return new observation of craft rv, fuel levels, system positions
        # give rewards for flying near other bodies, give big reward for reaching closed orbit around body
        if self._check_planetary_proximity():
            self.reward -= 10000
            return observation, self.reward, True, info
        self._calculate_rewards()

        self._check_if_done()

        # if self.done:
        #     logging.info(
        #         f"Current_Soi = {self.current_soi}, time = {self.current_time}, reward = {self.reward}, "
        #         f"remaining fuel = {self.spaceship.total_mass / self.spaceship.initial_mass}")

        return observation, self.reward, self.done, info

    def reset(self):
        # get planet and spaceship positions at start_time, reset spaceship fuel,

        self.current_time = self.start_time

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

    def _get_ephem(self):
        for body in self.body_dict["bodies"]:
            ephem = Ephem.from_horizons(body, self.current_time,
                                        attractor=self.body_dict["attractor"], id_type="majorbody")
            self.ephems[body] = ephem
        eph = get_body_barycentric_posvel([(6, 606)], self.current_time)
        pos = eph[0].get_xyz()
        vel = eph[1].get_xyz()

    def _get_observation(self):

        obs = []
        self._get_ephem()
        ship_r = self.spaceship.orbit.r.decompose().value / self.r_normalization.decompose().value
        ship_v = self.spaceship.orbit.v.decompose().value / self.v_normalization.decompose().value
        ship_obs = [
            self.spaceship.total_mass.decompose().value / self.spaceship.initial_mass.decompose().value,
            self.spaceship.propellant_mass.decompose().value / self.spaceship.initial_mass.decompose().value,
            *ship_r,
            *ship_v,
        ]
        obs.append(ship_obs)

        for i in self.ephems:
            body_obs = []
            body = moon_dict[i]["body"]
            body_r, body_v = self.ephems[i].rv()
            body_r = body_r.decompose().value / self.r_normalization.decompose().value
            body_v = body_v.decompose().value / self.v_normalization.decompose().value
            body_obs += [
                body.mass.decompose().value / Sun.mass.decompose().value,  # normalised by dividing by solar mass
                body.R.decompose().value / Sun.R.decompose().value,  # normalised by dividing by solar radius
                *body_r,
                *body_v
            ]
            obs.append(body_obs)
        np_obs = np.array(obs)
        return np_obs

    # def _record_current_state(self):
    #     # self.current_ephem
    #     # self.spaceship
    #     # write timestamp, orbits, and action
    #
    #     return

    def _calculate_rewards(self):
        # assign rewards for entering a planet's SoI (~=flybys), check if the probe is in a low orbit around a target

        # current_soi = self.spaceship.orbit.attractor
        # current_ecc = self.spaceship.orbit.ecc
        # previous_ecc = self.spaceship.previous_orbit.ecc
        # current_pericenter = self.spaceship.orbit.r_p
        # current_apocenter = self.spaceship.orbit.r_a
        #
        # if self.current_soi != max(self.visited_times.items(), key=lambda x: x[1])[0]:
        #     self.reward += 100
        # elif current_soi.name in self.target_bodies and current_ecc < previous_ecc:
        #     self.reward += 1
        # elif current_soi.name not in self.target_bodies and previous_ecc < current_ecc:
        #     self.reward += 1
        # else:
        #     self.reward -= 1
        #
        # if current_soi.name in self.target_bodies \
        #         and current_ecc > 0.5 \
        #         and current_apocenter < 1000 * u.km \
        #         and current_pericenter > current_soi.R:
        #     self.reward += 10000
        #     self.target_bodies.remove(current_soi.name)
        # if in orbit around a target, assign reward and then remove current system as target

        return

    def _check_if_done(self):
        # check if no target_bodies exist
        # if done, adjust reward based off elapsed time and remaining fuel
        if self.current_time > self.finish_time:
            self.done = True
            self.reward -= 10
        if not self.target_bodies:
            self.done = True
            self.reward += 10000
        if self.spaceship.propellant_mass == 0:
            self.done = True
            self.reward -= 10
            # (self.start_time - self.current_time)  add something to provide incentive for finishing quickly?
        return

    def _calculate_action_delta_v(self, action):
        *direction, thrust_percent = action
        thrust_percent = (thrust_percent / 2) + 0.5
        direction = direction / np.linalg.norm(direction)
        thrust_time = self.simulation_step * thrust_percent
        exhaust_velocity = self.spaceship.isp * g0
        mass_flow_rate = self.spaceship.thrust / exhaust_velocity
        delta_m = mass_flow_rate * thrust_time
        mass_start = self.spaceship.total_mass
        mass_final = mass_start - delta_m
        if mass_final > self.spaceship.dry_mass:
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
        self.spaceship.orbit = self.spaceship.orbit.apply_maneuver(maneuvers)

    def _calculate_laplace_radii(self):
        for i in self.body_dict["bodies"]:
            body = moon_dict[i]["body"]
            a = moon_dict[i]["orbital_radius"]
            r_soi = a * (body.k / body.parent.k) ** (2 / 5)
            moon_dict[i]["soi"] = r_soi

    # def _update_current_soi(self):
    #     if self.current_soi == self.body_dict["attractor"].name:
    #         for body in self.body_dict["bodies"]:
    #             body_r = ICRS(self.ephems[body].rv()[0])\
    #                 .transform_to(self.spaceship.orbit.get_frame())\
    #                 .represent_as(CartesianRepresentation)\
    #                 .xyz
    #             distance = norm(self.spaceship.orbit.r - body_r)
    #             if distance < moon_dict[body]["soi"]:
    #                 self._change_attractor(body)
    #                 break
    #     else:
    #         if self.spaceship.orbit.r > moon_dict[self.current_soi]:
    #             self._change_attractor(self.body_dict["attractor"].name)
    #
    # def _change_attractor(self, body):
    #     if body == self.body_dict["attractor"].name:
    #         pass
    #     else:
    #         pass

    def _check_planetary_proximity(self):

        attractor_r = self.spaceship.orbit.attractor.R.to(u.km).value
        orbit_periapsis = self.spaceship.orbit.r_p.to(u.km).value

        if orbit_periapsis < attractor_r:
            ship_altitude = self.spaceship.orbit.a.to(u.km).value
            ship_distance_travelled = (np.linalg.norm(self.spaceship.orbit.v) * self.action_step).to(u.km).value

            if ship_altitude < ship_distance_travelled:
                return True
        return False


class SpaceShip:

    def __init__(self, *, orbit, dry_mass, propellant_mass, isp, thrust):
        self.orbit = orbit
        self.previous_orbit = orbit
        self.dry_mass = dry_mass
        self.propellant_mass = propellant_mass
        self.isp = isp
        self.thrust = thrust
        self.initial_mass = dry_mass + propellant_mass

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
