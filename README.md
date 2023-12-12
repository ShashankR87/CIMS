# Cooperative Intersection Maneuvering System

Project Team:
- Marta Taulet
- Shashank Ramesh

This project addresses the challenge of intersection traffic management through the implementation of a central processing unit strategically positioned at intersections, effectively replacing traditional traffic lights. The system comprises two key components: Smart Vehicles and the Processing Unit.

A Smart Vehicle, refers to any vehicle equipped with the capability to communicate with the Processing Unit, facilitating the registration of its data and compliance with the received control signals. 

The Processing Unit serves as a centralized arbiter, collating data from multiple vehicles approaching the intersection. Utilizing this information, the Processing Unit then issues control signals to the vehicles, ensuring that if followed, each vehicle can navigate the intersection without collisions and in an efficient manner, ultimately reaching its designated destination.

The proposed system is visually represented below, illustrating the seamless interaction between Smart Vehicles and the Processing Unit, contributing to enhanced intersection traffic management.

<p align="center">
  <img width="460" height="300" src="https://github.com/ShashankR87/CIMS/assets/34104519/f79fd649-3d6a-4762-908f-af2d232a8dfa">
</p>

# System Dependencies
To run the simulation, your environment needs to have Python 3.7 or higher.

# Instructions

1. Install all the dependencies

`pip install -e requirements.txt`

3. Run the project

`python3 controller.py`

5. Optional arguments
   
--saved: You can provide a path to a preexisting model so the RL algorithm can resume the training with the provided weights instead of randomly initialized ones.

--train: Determines whether you want the model to be trained or to do inference.

`python3 controller.py --saved <path for model file> --train <True/False>`

### Simulation Environment

We opted for CARLA, an open-source simulator designed for autonomous driving research, as our primary environment. CARLA utilizes the Unity game engine and encompasses all the essential mechanics and assets crucial for our project. In our initial configuration, we established a four-way intersection as the focal point of our simulation. The environment is populated with up to four randomly initialized cars, each positioned on a different street leading to this intersection.
Each vehicle is also assigned the following initial attributes:

1. An initial velocity that is randomly sampled from a uniform distribution ranging from 5 m/s to 20 m/s.
2. A fixed route leading to a destination starting from the initial position. This route is represented as a series of waypoints in the CARLA environment.

The simulation begins with the cars navigating towards the intersection based on their initial configurations. The simulation ends under the following conditions: (1) a collision occurs, (2) one or more cars fail to reach their destinations within a predefined time threshold (set at 20 simulation seconds), or (3) all cars successfully reach their designated destinations. An example of an initial scene is shown in the image below on the left, depicting three cars approaching the intersection. The image on the right shows the same cars, a few seconds later, in the process of following their respective paths.

<p align="center">
  <img width="460" height="300" src="https://github.com/ShashankR87/CIMS/assets/34104519/5a44be7c-4d4a-42ca-8835-540f0443daaf">
  <img width="460" height="300" src="https://github.com/ShashankR87/CIMS/assets/34104519/57726db7-4b30-476d-bd0c-2c8d331d2dcd">
</p>

### The Processing Unit

The Processing Unit is engineered around a sophisticated deep reinforcement learning model, designed to process incoming vehicle data. This data encompasses the initial position, velocity, and desired destination of each Smart Vehicle. The pivotal choice for our project was the Proximal Policy Optimization (PPO) algorithm, a cutting-edge framework recognized for its efficacy in handling reinforcement learning tasks.

The proposed control system operates within the following specifications:

- Observation Space:
[Initial position, Initial velocity, Destination location] for each car in the intersection.

- Action Space: [Time (in seconds) for the car to reach the intersection entry] for each car in the intersection.


The PPO model architecture was fine-tuned, incorporating additional dense layers to enhance the processing of input data. The resulting structure is represented
in the simplified diagram below.

<p align="center">
  <img width="800" height="300" src="https://github.com/ShashankR87/CIMS/assets/34104519/284c6e58-2f09-43e0-b587-5cffca724a13">
</p>

An important part of our reinforcement learning setup was the reward function. The reward policy dictated the assignment of rewards or penalties to the PPO model after each simulation based on its performance. It was defined as follows:

• -1000 if there was a collision

• +1000 / (total journey time) for each car that successfully reached its destination

• -10 * (distance from the destination) for each car that did not completely traverse its path in time

### Results

Our initial project configuration was structured as follows:

We considered a four-way intersection scenario involving three vehicles, each with the option to follow one of three distinct paths (left, right, center). Additionally, each car was assigned an initial velocity randomly within a specified range. The graphs for collision per episode and reward per episode are shown below.

<p align="center">
  <img width="460" height="300" src="https://github.com/ShashankR87/CIMS/assets/34104519/7d30f4c0-cf7d-4040-a5d5-89078dcfd82c">
  <img width="460" height="300" src="https://github.com/ShashankR87/CIMS/assets/34104519/fe56b40e-111a-48aa-ab0a-d2e5bfd2a1a6">
</p>




