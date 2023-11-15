import carla 
import glob
import os
import random 
import sys
import time
import numpy as np
import numpy as np
import keras.backend.tensorflow_backend as backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2

# TODO: Create requirements.txt

N_CARS = 2
VEHICLE_MODEL = 'vehicle.jeep.wrangler_rubicon'
A_MIN = 0
A_MAX = 10
ACTION_SPACE_SIZE = 10
T = 20

ACC_MAP = {
    0: -10,
    1: -9,
    2: -8,
    3: -7,
    4: -6,
    5: -5,
    6: -4,
    7: -3,
    8: -2,
    9: -1,
    10: 0,
    11: 1,
    12: 2,
    13: 3,
    14: 4,
    15: 5,
    16: 6,
    17: 7,
    18: 8,
    19: 9
}
'''
Paths to make car move through the intersection (Direction is relative to the fixed spectator pov)
N->E (49, 27)
N->S (49, 3)
N->W (50, 67)

E->N (28, 57)
E->S (137, 77)
E->W (28, 95)

S->E (85, 26)
S->N (80, 56)
S->W (80, 95)

W->S (138, 2)
W->E (139, 27)
W->N (139, 56)
'''


class CarlaEnv:

    def __init__(self):
        try:
            sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
        except IndexError:
            # TODO: Error handling
            pass

        self.client = carla.Client('localhost', 2000) 
        self.world = self.client.get_world()
        self.bp_lib = self.world.get_blueprint_library() 
        self.spawn_points = self.get_spawn_points()
        self.traffic_manager = self.client.get_trafficmanager(2000)  # Making it class attribute so we can use it in multiple places
        self.vehicles = []

        # settings = world.get_settings()
        # settings.synchronous_mode = True # Enables synchronous mode
        # # settings.fixed_delta_seconds = 0.05
        # world.apply_settings(settings)


    def spectator_setup(self, world, x, y, z):
        spectatorLoc = carla.Location(x, y, z)
        spectator = world.get_spectator() 
        transform = carla.Transform(spectatorLoc, carla.Rotation(-90)) 
        spectator.set_transform(transform)


    def get_spawn_points(self):
        # get spawn points and label them on the map
        spawn_points = self.world.get_map().get_spawn_points()
        for i, spawn_point in enumerate(spawn_points):
            self.world.debug.draw_string(spawn_point.location, str(i), life_time=1000)
        return spawn_points


    def spawn_vehicle(self, spawn_point):
        vehicle_bp = self.bp_lib.find(VEHICLE_MODEL)
        vehicle = self.world.spawn_actor(vehicle_bp, self.spawn_points[spawn_point])

        return vehicle

    def detect_collision(self, vehicle, collision_box):
        if  ((vehicle.get_location().x >= collision_box.location.x - 1) and \
            (vehicle.get_location().x <= collision_box.location.x + 1)) and \
            ((vehicle.get_location().y >= collision_box.location.y - 1) and \
            (vehicle.get_location().y <= collision_box.location.y + 1)):
                print('reached')
                print(str(vehicle.get_location().x), ' ', str(vehicle.get_location().y))
                return True
        return False
    

    def control_manual(self, vehicle, spawn_point_collision):
        vehicle.apply_control(carla.VehicleControl(throttle = 1.0))

        counter = 0
        while True:
            counter += 1
        #     if counter % 1000 == 0:
        #         print(str(vehicle1.get_location().x) + ' ' + str(vehicle1.get_location().y))
            if self.detect_collision(vehicle, self.spawn_points[spawn_point_collision]):
                vehicle.apply_control(carla.VehicleControl(hand_brake = True))
                break

        time.sleep(50)


    def control_traffic_manager(self, vehicle, a, b):
        route = [self.spawn_points[a].location, self.spawn_points[b].location]

        # Set all traffic lights in the map to green
        list_actor = self.world.get_actors()
        for actor_ in list_actor:
            if isinstance(actor_, carla.TrafficLight):
                actor_.set_state(carla.TrafficLightState.Green) 
                actor_.set_green_time(1000.0)

        vehicle.set_autopilot(True)

        self.traffic_manager.update_vehicle_lights(vehicle, True)
        self.traffic_manager.random_left_lanechange_percentage(vehicle, 0)
        self.traffic_manager.random_right_lanechange_percentage(vehicle, 0)
        self.traffic_manager.auto_lane_change(vehicle, False)
        self.traffic_manager.ignore_lights_percentage(vehicle, 100)
        self.traffic_manager.ignore_vehicles_percentage(vehicle, 100)
        self.traffic_manager.ignore_signs_percentage(vehicle, 100)

        self.traffic_manager.set_path(vehicle, route)


    def spawn_n_cars(self):
        for i in N_CARS:
            path = None          # TODO: Shashank
            
            carla_car = self.spawn_vehicle(path.start_point)
            acc =  np.random.randint(A_MIN, A_MAX)
            vel = acc * T
            vehicle_data = {"carla_car": carla_car, 
                            "path": path, 
                            "acc": acc,
                            "vel": vel}
            
            self.vehicles.append(vehicle_data)

    def action(self, acc_list):
        '''The car can take any of the ACTION_SPACE_SIZE possible throttle values from A_MIN to A_MAX'''

        for i, car_dict in enumerate(self.vehicles):
            throttle = np.clip(acc_list[i] / ACTION_SPACE_SIZE, 0, 1)
            act = carla.VehicleControl(throttle=float(throttle))
            car_dict["carla_car"].apply_control(act)

    def destroy_all_cars(self):
        for car in self.vehicles:
            car["carla_car"].destroy()


    def reset(self):
        self.destroy_all_cars()
        self.spawn_n_cars()
        self.episode_step = 0
        return  

    def step(self, acc_list):
        '''Apply respective acceleration for each car'''

        self.episode_step += 1
        self.action(acc_list)

        if collision:
            reward = -self.ENEMY_PENALTY

        elif car1 reaches destination:
            reward = self._REWARD
        

        



def main():

    # carla setup
    carla_env = CarlaEnv()

    # fix spectator on top of intersection
    carla_env.spectator_setup(x=-47, y=16.8, z=60)

    agent = DQN()

    run_episodes(agent)

    # Control car by varying throttle manually
    carla_env.control_manual(vehicle, spawn_point_collision=79)
    vehicle.destroy()

    # Control car using traffic manager with custom path
    vehicle = carla_env.spawn_vehicle(spawn_point=50)
    carla_env.control_traffic_manager(vehicle, a=50, b=67)
    vehicle.destroy()


if __name__ == "__main__":
    main()