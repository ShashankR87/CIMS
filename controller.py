from datetime import datetime
import carla 
import glob
import os
import random 
import sys
import time
import numpy as np
import numpy as np
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import threading
import math
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import gym
from gym.spaces import Box, MultiDiscrete
from gym import spaces
from gym.envs.registration import register
import cmath
import argparse
import wandb

# Environment settings
FIXED_TIME_STEP = 0.05
COLLISION_PENALTY = 1_000
DESTINATION_REWARD = 1_000
DID_NOT_REACH_PENALTY = 10
N_CARS = 3
VEHICLE_MODEL = 'vehicle.jeep.wrangler_rubicon'
T = 20

ACC_MAP = {i: i for i in range(10)}   # {0: -10, 1: -9, ... 18: 8, 19: 9}
ACTION_SPACE_SIZE = len(ACC_MAP)           # 20
V_MIN = 15.0
V_MAX = 20.0
MAX_ALIVE_TIME = 100

TRAJECTORIES = [                                            
    (139, [390, 302, 298, 431, 429, 508, 435, 495, 343, 351]),   #W->N
    (138, [389, 303, 299, 430, 426, 424, 666, 664]),             #W->E
    (138, [389, 303, 299, 501, 500, 327, 323, 315]),             #W->S
    (50, [353, 349, 345, 341, 487, 262, 336, 296, 300, 392]),    #N->W
    (50, [353, 349, 345, 341, 334, 327, 323, 315]),              #N->S
    (49, [352, 348, 344, 340, 436, 435, 492, 423, 665, 663]),    #N->E
    (28, [671, 673, 328, 472, 473, 338, 342 , 346, 350]),        #E->N
    (28, [671, 673, 328, 332, 336, 300, 392]),                   #E->W
    (29, [672, 674, 329, 289, 493, 508, 482, 318, 314]),         #E->S
    (85, [312, 316, 488, 490, 426, 424, 287, 666, 664]),         #S->E
    (85, [312, 316, 488, 492, 494, 342, 346, 350]),              #S->N
    (80, [313, 317, 489, 491, 427, 82, 512, 337, 301, 391])      #S->W
]


class CarlaEnv:

    def __init__(self):
        try:
            sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
        except IndexError:
            pass

        self.client = carla.Client('localhost', 2000) 
        self.world = self.client.get_world()
        self.bp_lib = self.world.get_blueprint_library()
        self.spawn_points = self.get_spawn_points()
        self.waypoints = self.world.get_map().generate_waypoints(10)
        self.vehicles = []
        self.collision = False
        self.sim_time = 0
        self.reward = 0
        self.total_times = []
        self.action_space = Box(low=1.0, high=5.0, shape=(N_CARS,), dtype=np.int32)

        spawn_locations = [(s.location.x, s.location.y) for s in self.spawn_points]
        min_x = min(spawn_locations, key= lambda x: x[0])[0]
        max_x = max(spawn_locations, key= lambda x: x[0])[0]
        min_y = min(spawn_locations, key= lambda x: x[1])[1]
        max_y = max(spawn_locations, key= lambda x: x[1])[1]

        self.observation_space = spaces.Box(low=np.array(([V_MIN, min_x, min_y, min_x, min_y]) * N_CARS), high=np.array(([V_MAX, max_x, max_y, max_x, max_y]) * N_CARS), dtype=np.float32)
        
        
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = FIXED_TIME_STEP
        self.world.apply_settings(settings)

        self.spectator_setup(x=-47, y=16.8, z=100)

    def __getstate__(self):
        # Return None to exclude the client object from pickling
        return {
            'action_space': self.action_space,
            'observation_space': self.observation_space
        }
    
    def __setstate__(self, state):
        self.action_space = state['action_space']
        self.observation_space = state['observation_space']

    def spectator_setup(self, x, y, z):
        spectatorLoc = carla.Location(x, y, z)
        spectator = self.world.get_spectator() 
        transform = carla.Transform(spectatorLoc, carla.Rotation(-90)) 
        spectator.set_transform(transform)

    def get_env_data(self):
        if len(self.total_times) > 0:
            total_times = self.total_times
        else:
            total_times = [car["total_time"] for car in self.vehicles]
        avg_total_time = sum(total_times) / len(total_times)

        return self.reward, self.collision, avg_total_time
    def get_spawn_points(self, show=False):
        # get spawn points and label them on the map
        spawn_points = self.world.get_map().get_spawn_points()
        if show:
            for i, spawn_point in enumerate(spawn_points):
                self.world.debug.draw_string(spawn_point.location, str(i), life_time=1000)
        return spawn_points


    def spawn_vehicle(self, spawn_point):

        vehicle_bp = self.bp_lib.find(VEHICLE_MODEL)
        
        car_color = list(np.random.choice(range(256), size=3))
        car_color = [str(c) for c in car_color]
        car_color = ','.join(car_color)
        vehicle_bp.set_attribute('color', car_color)
        vehicle = self.world.spawn_actor(vehicle_bp, self.spawn_points[spawn_point])
        return vehicle

    def handle_collision(self, event):
        self.collision = True
        
    def create_path(self, trajectory):
        path = []
        waypoints = [self.waypoints[w].transform.location for w in trajectory]
        for waypoint in waypoints:
            path.append(carla.Location(x=waypoint.x, y=waypoint.y, z=1))
        return path


    def spawn_n_cars(self):
        # paths_to_choose = [0,4,7,10] #For Experiment 2
        chosen_directions = []
        for i in range(N_CARS):
            direction = random.randint(0, 3)
            while direction in chosen_directions:
                direction = random.randint(0, 3)
            chosen_directions.append(direction)
            trajectory = random.choice(TRAJECTORIES[direction*3: direction*3 +3])
            '''
            # For Experiment 2
            trajectory = TRAJECTORIES[paths_to_choose[i]]

            direction = random.randint(0, 3)
            while direction in chosen_directions:
                direction = random.randint(0, 3)
            chosen_directions.append(direction)
            trajectory = TRAJECTORIES[paths_to_choose[direction]]
            '''
            path = self.create_path(trajectory[1])

            car_actor = self.spawn_vehicle(trajectory[0])
            collision_sensor_bp = self.bp_lib.find("sensor.other.collision")
            collision_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
            collision_sensor = self.world.spawn_actor(collision_sensor_bp, collision_transform, attach_to=car_actor)
            collision_sensor.listen(lambda event: self.handle_collision(event))

            velocity = random.randint(V_MIN, V_MAX)
            
            vehicle_data = {
                "carla_car": car_actor, 
                "path": path,
                "path_idx": TRAJECTORIES.index(trajectory),
                "velocity": velocity,
                "collision_sensor": collision_sensor,
                "total_time": 0,
                "start_index": 0,
                "location_x": self.spawn_points[trajectory[0]].location.x,
                "location_y": self.spawn_points[trajectory[0]].location.y,
                "start_time": datetime.now()
            }
            
            self.vehicles.append(vehicle_data)


    def calculate_yaw_angle_to_target(self, current_location, target_location):
        direction_vector = target_location - current_location
        angle_radians = math.atan2(direction_vector.y, direction_vector.x)
        angle_degrees = math.degrees(angle_radians)
        return angle_degrees

    # Function to orient the vehicle towards a target point
    def orient_vehicle_towards_target(self, vehicle, target_point):
        current_location = vehicle.get_location()
        yaw_angle = self.calculate_yaw_angle_to_target(current_location, target_point)
        vehicle.set_transform(carla.Transform(current_location, carla.Rotation(yaw=yaw_angle)))

    def calculate_target_direction_degrees(self, start_point, end_point):
        delta_x = end_point.x - start_point.x
        delta_y = end_point.y - start_point.y

        # Calculate the angle in radians using arctangent
        angle_radians = math.atan2(delta_y, delta_x)

        # Convert radians to degrees
        angle_degrees = math.degrees(angle_radians)

        # Ensure the angle is in the range [0, 360)
        angle_degrees = (angle_degrees + 360) % 360

        return angle_degrees
    
    def calculate_distance(self, start_point, end_point):
        delta_x = end_point.x - start_point.x
        delta_y = end_point.y - start_point.y
        return math.sqrt(delta_x ** 2 + delta_y ** 2)


    def calculate_time_with_acceleration(self, acceleration, initial_velocity, initial_displacement):
        a = 1
        b = 2 * initial_velocity / acceleration
        c = -2 * initial_displacement / acceleration

        # Calculate the discriminant
        if b**2 - 4*a*c < 0:
            return -1
        discriminant = (b**2 - 4*a*c)**0.5

        # Calculate the two possible solutions for time
        t1 = (-b + discriminant) / (2 * a)
        t2 = (-b - discriminant) / (2 * a)
        t = min(t1, t2)

        if t < 0:
            return t2
        return t1
    
    def follow_path_with_velocity(self, vehicle_data, car_idx, time_to_intersection):
        cnt = 0
        vehicle = vehicle_data["carla_car"]
        path = vehicle_data["path"]
        velocity = vehicle_data["velocity"]
        distance_to_intersection = self.calculate_distance(path[1], path[2])
        new_velocity = distance_to_intersection / (time_to_intersection)
        print(f"init vel is {velocity} and new vel is {new_velocity}")
        
        start_time = datetime.now()
        for location in path:
            direction = location - vehicle.get_location()
            target_direction_degrees = self.calculate_target_direction_degrees(vehicle.get_location(), location)
            target_direction_radians = math.radians(target_direction_degrees)
            
            self.orient_vehicle_towards_target(vehicle, location)
            current_angle = self.calculate_yaw_angle_to_target(vehicle.get_location(), location)
            if cnt > 1:
                velocity = new_velocity
            
            target_velocity_vector = carla.Vector3D(x=math.cos(target_direction_radians), y=math.sin(target_direction_radians), z=0.0) * velocity
            vehicle.set_target_velocity(target_velocity_vector)
            cnt += 1
            while vehicle.is_alive:
                if not vehicle.is_alive:
                    return
                new_angle = self.calculate_yaw_angle_to_target(vehicle.get_location(), location)
                if abs(new_angle - current_angle) >= 90:
                    break
                if (datetime.now() - self.vehicles[car_idx]["start_time"]).total_seconds() > MAX_ALIVE_TIME * FIXED_TIME_STEP:
                    self.destroy_actor(vehicle_data)
                    return
        
        self.vehicles[car_idx]["total_time"] = (datetime.now() - start_time).total_seconds()
        self.destroy_actor(vehicle_data)

    def action(self, time_list):
        thread_list = []
        for i, car_dict in enumerate(self.vehicles):
            thread_list.append(threading.Thread(target=self.follow_path_with_velocity, args=(car_dict, i , time_list[i])))
        for t in thread_list:
            t.start()
        for t in thread_list:
            t.join()
        
        self.destroy_all_actors()
        

    def destroy_all_actors(self):
        for car in self.vehicles:
            if car["carla_car"].is_alive:
                car["carla_car"].destroy()
            if car["collision_sensor"].is_alive:
                car["collision_sensor"].destroy()
    def destroy_actor(self, car):
        if car["carla_car"].is_alive:
            car["carla_car"].destroy()
        if car["collision_sensor"].is_alive:
            car["collision_sensor"].destroy()

    def convert_to_onehot(self, path_idx):
        onehot = [0] * len(TRAJECTORIES)
        onehot[path_idx] = 1
        return onehot

    def reset(self):
        total_times = [car["total_time"] for car in self.vehicles]
        self.total_times  = total_times
        self.destroy_all_actors()
        self.vehicles = []
        self.spawn_n_cars()
        self.episode_step = 0
        self.collision = False
       
        state = []
        for v in self.vehicles:
            state.append(v["velocity"])
            state.append(v["location_x"])
            state.append(v["location_y"])
            state.append(v["path"][-1].x)
            state.append(v["path"][-1].y)
        return np.array(state)


    def step(self, time_list):

        self.episode_step += 1
        self.action(time_list)

        if self.collision:
            reward = -COLLISION_PENALTY
            state = []
            for v in self.vehicles:
                state.append(v["velocity"])
                state.append(v["location_x"])
                state.append(v["location_y"])
                state.append(v["path"][-1].x)
                state.append(v["path"][-1].y)
            
            print('Reward for episode: ', str(reward))
            self.reward = reward
            return state, reward, True, {}

        else:
            # # the reward for completion is inversely proportional to the total time it took to traverse the intersection
            reached_destination = [car for car in self.vehicles if car["total_time"]>0]
            rewards_reached_destination = [DESTINATION_REWARD * (FIXED_TIME_STEP / car["total_time"]) for car in reached_destination]
            reward = sum(rewards_reached_destination)
            did_not_reach = [car for car in self.vehicles if car["total_time"]<=0]
            did_not_reach_penalty = [DID_NOT_REACH_PENALTY * (len(v["path"]) - v["start_index"]) for v in did_not_reach]
            reward -= sum(did_not_reach_penalty)
            state = []
            for v in self.vehicles:
                state.append(v["velocity"])
                state.append(v["location_x"])
                state.append(v["location_y"])
                state.append(v["path"][-1].x)
                state.append(v["path"][-1].y)
            
            print('Reward for episode: ', str(reward))
            self.reward = reward
            return np.array(state), reward, True, {}
    

class CustomWrapper(gym.Env):
    def __init__(self, your_custom_env):
        self.env = your_custom_env
        self.action_space = your_custom_env.action_space
        self.observation_space = your_custom_env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)
    def destroy(self):
        self.env.destroy_all_actors()
    def get_env_data(self):
        return self.env.get_env_data()

class MyCallback(BaseCallback):
    def __init__(self, verbose=0):

        super(MyCallback, self).__init__(verbose)

        self.reward_per_episode = 0
        self.episode_collision = 0
        self.episode_total_time = 0
        self.num_steps = 0


    def _on_step(self):
        reward, collision, total_time = self.training_env.env_method(method_name='get_env_data')[0]
        
        self.reward_per_episode += reward
        
        self.episode_collision += 1 if reward == -COLLISION_PENALTY else 0
        self.episode_total_time += total_time
        self.num_steps += 1

        return True
    
    def _on_rollout_end(self) -> None:
        # This method will be called at the end of each episode
        print(f"Num timesteps: {self.num_steps}, \
            episode reward: {self.reward_per_episode}, \
              average total time: {self.episode_total_time / self.num_steps}, \
              collision: {self.episode_collision}")

        # log metrics to wandb
        wandb.log({"reward_per_episode": self.reward_per_episode, 
                   "average_time_taken": self.episode_total_time / self.num_steps,
                   "episode_collision": self.episode_collision})
        
        self.reward_per_episode = 0  # Reset episode reward for the next episode
        self.episode_total_time = 0
        self.num_steps = 0
        self.episode_collision = 0

def main():

    argparser = argparse.ArgumentParser()
    
    argparser.add_argument(
        '--saved',
        metavar='s',
        default='None',
        help='Filename/path to a saved model file',
        nargs='?')
    
    argparser.add_argument(
        '--train',
        metavar='t',
        default='True',
        help='Trains the model before inference', 
        nargs='?')

    args = argparser.parse_args()


    print("Execution started")
    n_steps = 128
    train_time = 1_000
    num_iter = 5

    wandb.init(
        project="cims",
        config={
            "n_steps": n_steps,
            "total_train_time": train_time * num_iter,
        }
    )
    

    if not os.path.isdir('models'):
        os.makedirs('models')
    if not os.path.isdir('logs'):
        os.makedirs('logs')

    register(
    id='CustomWrapped-v0',
    entry_point='__main__:CustomWrapper',
    kwargs={'your_custom_env': CarlaEnv()},
    )
    wrapped_env = gym.make('CustomWrapped-v0')
    callback = MyCallback()

    policy_kwargs = dict(net_arch=dict(pi=[128, 256, 64], vf=[128, 256, 64]))
    model = PPO("MlpPolicy", wrapped_env, verbose=10, policy_kwargs=policy_kwargs, tensorboard_log="logs", n_steps=n_steps)
    start_iter = 0
    if args.saved != 'None':
        start_iter = int(args.saved.split('_')[2]) + 1
        model.load(args.saved)
    if args.train == 'True':
        for i in range(start_iter, start_iter + num_iter):
            model.learn(total_timesteps=train_time, progress_bar=True, callback = callback, tb_log_name="Cims_v13", reset_num_timesteps= False)
            current_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            model.save(f"models/cims_model_{i}_{current_timestamp}")

    print('Done')
    wrapped_env.destroy()
    wandb.finish()



if __name__ == "__main__":
    main()