from datetime import datetime
from enum import Enum
from typing import Tuple, List, Dict

import gym
from astropy.units import Quantity
from astropy.time import Time, TimeDelta
from gym import spaces
from astropy.coordinates import solar_system_ephemeris
from astropy import time, units as u
from astropy.constants import g0
from poliastro.bodies import Earth, Sun, Moon, Mercury, Venus, Mars, Jupiter, Saturn, Uranus, Neptune, \
    SolarSystemPlanet
from poliastro.ephem import Ephem
from poliastro.maneuver import Maneuver
import numpy as np
from poliastro.frames import Planes
from poliastro.twobody.orbit import Orbit
from poliastro.threebody.soi import laplace_radius
from poliastro.twobody.events import LithobrakeEvent
from poliastro.twobody.propagation import propagate, cowell


class SpaceShipName(Enum):
    DEFAULT = "default, approximately Cassini-Huygens with increased dV (4600) to compensate for lack of 3rd stage"
    LOW_THRUST = "low_thrust, approximately Deep Space 1 with ~6500 m/s dV"
    HIGH_THRUST = "high_thrust"


class SystemScope(Enum):
    EARTH = "Earth"
    PLANETS = "Sun and Planets"
    # add more systems?


class SolarSystem(gym.Env):
    metadata = {'render.modes': ['human']}

    @u.quantity_input
    def __init__(self,
                 bodies: SystemScope = SystemScope.PLANETS,
                 start_body: SolarSystemPlanet = None,
                 target_bodies: List[SolarSystemPlanet] = None,
                 start_time: Time = None,
                 action_step: u.s = 3600 * u.s,
                 simulation_ratio: int = 60,
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
            # possible todo: specify whether to orbit or fly by planet?
        if start_time is None:
            start_time = Time(datetime.now()).tdb

        # enforce action_step/simulation_step is an integer?
        if action_step.value % simulation_ratio != 0:
            raise ValueError("Action step must be evenly divisible by simulation_ratio")

        self.start_body = start_body
        self.target_bodies = []
        for target in target_bodies:
            self.target_bodies.append(target.name)
        self.spaceship_initial_altitude = spaceship_initial_altitude
        self.start_time = start_time
        self.current_time = None
        self.action_step = action_step
        self.simulation_ratio = simulation_ratio
        self.done = False
        self.reward = 0
        self.initial_reset = False

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
            SystemScope.PLANETS:
                [Sun, Earth, Mercury, Venus, Mars, Jupiter, Saturn, Uranus, Neptune]
        }
        # define bodies to model
        # poliastro.bodies.SolarSystemPlanet =
        #   Sun, Earth, Moon, Mercury, Venus, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto
        # could also add versions for: only inner solar system, only 'major' bodies jovan moons, saturn's moons?

        try:
            self.body_list = body_dict[bodies]
        except KeyError:
            raise KeyError(f"bodies must be one of {body_dict.keys()}")

        self.soi_radii = self._calculate_system_laplace_radii()
        self.current_soi = self.start_body.name
        self.visited_times = {self.start_body.name: Time(datetime.now()).tdb}

        # set up spacecraft

        self.spaceship = self._init_spaceship()

        self.current_ephem = None

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10, 11))

        # action:
        # tuple [[x,y,z], burn duration]
        self.action_space = spaces.Tuple((
            spaces.Box(low=-1.0, high=1.0, shape=(3,)),  # x,y,z direction vector
            spaces.Box(low=0.0, high=1.0, shape=(1,))  # burn duration as percent of time_step
        ))

    @property
    def simulation_step(self):
        return self.action_step / self.simulation_ratio

    def step(self, action):

        info = []

        if not self.initial_reset:
            return self.reset(), self.reward, self.done, info

        dv = self._calculate_action_delta_v(action)
        # take action in the form of a direction and burn time fraction, output total delta v
        maneuvers = self._split_burn_to_impulse_list(dv)
        # split the delta v into impulses for each simulation step to smooth the impulse. returns list of impulses
        self._apply_maneuvers(maneuvers)
        # apply the impulses for maneuver, including propagating the orbit forwards to the end of that
        self.current_time = self.spaceship.orbit.epoch
        self.current_ephem = self._get_ephem_from_list_of_bodies(self.body_list, self.current_time)
        self._update_current_soi()

        observation = self._get_observation()

        self._record_current_state()

        # return new observation of craft rv, fuel levels, system positions
        # todo: calculate rewards? other info?
        # give rewards for flying near other bodies, give big reward for reaching closed orbit around body
        if self._check_for_lithobraking():
            return observation, -100, True, info
        self._calculate_rewards()
        # when target is visited to within desired thresholds, mark it as visited.
        # when all targets are done, set done = True
        self._check_if_done()

        return observation, self.reward, self.done, info

    def reset(self):
        # get planet and spaceship positions at start_time, reset spaceship fuel,

        self.current_time = self.start_time

        # get ephem
        self.current_ephem = self._get_ephem_from_list_of_bodies(self.body_list, self.start_time)

        # set up spacecraft
        self.spaceship = self._init_spaceship()

        observation = self._get_observation()

        self.initial_reset = True

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
        pass

    @staticmethod
    def _get_ephem_from_list_of_bodies(bodies, current_time) -> Dict[str, Tuple[SolarSystemPlanet, Ephem]]:
        list_of_bodies = {}
        for i in bodies:
            ephem = Ephem.from_body(i, current_time)
            list_of_bodies[i.name] = (i, ephem)
        return list_of_bodies

    def _get_observation(self):

        obs = []
        attractor_r, attractor_v = self.current_ephem[self.spaceship.orbit.attractor.name][1].rv()
        ship_r = attractor_r.decompose().value + self.spaceship.orbit.r.decompose().value
        ship_v = attractor_v.decompose().value + self.spaceship.orbit.v.decompose().value
        # todo: check the above works, ie does not change frame weirdly
        ship_obs = [
            self.action_step.decompose().value,
            self.spaceship.thrust.decompose().value,
            self.spaceship.isp.decompose().value,
            self.spaceship.total_mass.decompose().value,
            self.spaceship.dry_mass.decompose().value,
            *ship_r[0],
            *ship_v[0]
        ]
        obs.append(ship_obs)

        for i in self.current_ephem:
            body_obs = []
            body = self.current_ephem[i][0]
            body_r, body_v = self.current_ephem[i][1].rv()
            body_r = body_v.decompose().value
            body_v = body_v.decompose().value
            if i in self.target_bodies:
                body_obs.append(True)
            else:
                body_obs.append(False)
            body_obs += [body.mass.decompose().value, body.R.decompose().value, *body_r[0], *body_v[0], 0, 0]
            obs.append(body_obs)
        np_obs = np.array(obs)
        # todo: normalisation ;_;
        # body mass/radius -> divide by solar mass/radius
        # body distance - find max body distance, hardcoded? speed find some measure for fastest planet?
        # ship mass -> fraction of total mass
        # ship thrust? isp?
        return np_obs

    def _record_current_state(self):
        # self.current_ephem
        # self.spaceship
        # write timestamp, orbits, and action

        return

    def _calculate_rewards(self):
        # assign rewards for entering a planet's SoI (~=flybys), check if the probe is in a low orbit around a target

        current_soi = self.spaceship.orbit.attractor
        current_ecc = self.spaceship.orbit.ecc
        current_pericenter = self.spaceship.orbit.r_p
        current_apocenter = self.spaceship.orbit.r_a

        if self.current_soi != max(self.visited_times.items(), key=lambda x: x[1])[0]:
            self.reward += 10

        if current_soi.name in self.target_bodies \
                and current_ecc > 0.5 \
                and current_apocenter < 1000 * u.km \
                and current_pericenter > current_soi.R:
            self.reward += 100
            self.target_bodies.remove(current_soi.name)
        # if in orbit around a target, assign reward and then remove current system as target

        return

    def _check_if_done(self):
        # check if no target_bodies exist
        # if done, adjust reward based off elapsed time and remaining fuel
        if not self.target_bodies:
            self.done = True
            self.reward += 100
            # (self.start_time - self.current_time)  add something to provide incentive for finishing quickly?
        return

    def _calculate_action_delta_v(self, action):
        direction, thrust_percent = action
        direction = direction / np.linalg.norm(direction)
        thrust_time = self.simulation_step * thrust_percent
        exhaust_velocity = self.spaceship.isp * g0
        mass_flow_rate = self.spaceship.thrust / exhaust_velocity
        delta_m = mass_flow_rate * thrust_time
        mass_start = self.spaceship.total_mass
        mass_final = mass_start - delta_m
        delta_v = exhaust_velocity * np.log((mass_start / mass_final))
        self.spaceship.total_mass = mass_final

        return delta_v * direction

    def _split_burn_to_impulse_list(self, dv):
        split_impulse = dv / self.simulation_ratio
        impulse = []
        for x in range(0, self.simulation_ratio):
            impulse.append((self.simulation_step, split_impulse))
        maneuvers = Maneuver(*impulse)
        return maneuvers

    def _apply_maneuvers(self, maneuvers):
        self.spaceship.orbit = self.spaceship.orbit.apply_maneuver(maneuvers)

    def _calculate_system_laplace_radii(self):
        body_soi = {}

        for body in self.body_list:
            if body.name == "Sun":
                pass
            else:
                try:
                    soi_rad = laplace_radius(body)

                    body_soi[body.name] = soi_rad
                except KeyError:
                    pass
                    # a = get_mean_elements(body).a <- semimajor axis
                    # r_SOI = a * (body.k / body.parent.k) ** (2 / 5)
                    # todo: calculate laplace radius from mass ratio and semimajor axis
                    #  for bodies where get_mean_elements fails
        return body_soi

    def _update_current_soi(self):
        if self.current_soi == Sun.name:
            for body in self.current_ephem:
                if self.soi_radii[body[0].name] > np.linalg.norm(self.spaceship.orbit.r - body[1].rv()[0]):
                    self.spaceship.orbit.change_attractor(body[0], force=True)
                    self.current_soi = body[0].name
                    self.visited_times[body[0].name] = self.current_time

        else:
            if np.linalg.norm(self.spaceship.orbit.r) > self.soi_radii[self.current_soi]:
                self.current_soi = Sun.name
                self.spaceship.orbit.change_attractor(Sun, force=True)
                # edit spacecraft orbit to be sun-based by adding spaceship rv to planet rv

    def _check_for_lithobraking(self):
        attractor_r = self.spaceship.orbit.attractor.R.to(u.km).value
        if self.spaceship.orbit.r_p.to(u.km).value < attractor_r:
            # if periapsis of orbit < diameter of orbit attractor
            lithobrake_event = LithobrakeEvent(attractor_r)
            events = [lithobrake_event]
            # make a new lithobrake event with that radius

            # todo: propagate orbit forward by time_step and check for lithobrake event
            # if lithobrake found, return true
            lithobrake = False
            if lithobrake:
                return True

        return False


class SpaceShip:

    def __init__(self, *, orbit, dry_mass, propellant_mass, isp, thrust):
        self.orbit = orbit
        self.dry_mass = dry_mass
        self.propellant_mass = propellant_mass
        self.isp = isp
        self.thrust = thrust
        self.rv = self.orbit.rv()  # type: Tuple[Quantity, Quantity]

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
            SpaceShipName.DEFAULT:
                SpaceShip(
                    orbit=start_orbit,
                    dry_mass=2500 * u.kg,
                    propellant_mass=10000 * u.kg,
                    # propellant mass is significantly increased to compensate for third stage
                    isp=300 * u.s,
                    thrust=500 * u.N
                    # approx dV = 4600
                ),
            SpaceShipName.HIGH_THRUST:
                SpaceShip(
                    orbit=start_orbit,
                    dry_mass=2500 * u.kg,
                    propellant_mass=10000 * u.kg,
                    isp=300 * u.s,
                    thrust=5000 * u.N
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
