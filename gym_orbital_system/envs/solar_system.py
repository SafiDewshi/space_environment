import gym
import poliastro
from gym import spaces
from astropy.coordinates import solar_system_ephemeris
from astropy import time
from poliastro.ephem import Ephem
import numpy as np


class SolarSystem(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, bodies="All", **kwargs):
        super(SolarSystem, self).__init__()
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
            "Earth": ["Earth", "Moon"],
            "All":
                ["Sun", "Earth", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"],
            "Inner": ["Sun", "Earth", "Moon", "Mercury", "Venus", "Mars"],
            "Major": ["Sun", "Earth", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
        }

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


