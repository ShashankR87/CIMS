import carla 
import glob
import logging
import math 
import os
import random 
import sys
import time

# TODO: Make Requirements.txt

VEHICLE_MODEL = 'vehicle.jeep.wrangler_rubicon'

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


class CarlaWorld:

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

def main():

    # carla setup
    carla_world = CarlaWorld()

    # fix spectator on top of intersection
    carla_world.spectator_setup(x=-47, y=16.8, z=60)

    vehicle = carla_world.spawn_vehicle(spawn_point=28)

    # Control car by varying throttle manually
    carla_world.control_manual(vehicle, spawn_point_collision=79)
    vehicle.destroy()

    # Control car using traffic manager with custom path
    vehicle = carla_world.spawn_vehicle(spawn_point=50)
    carla_world.control_traffic_manager(vehicle, a=50, b=67)
    vehicle.destroy()


if __name__ == "__main__":
    main()