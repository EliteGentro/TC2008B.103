# %pip install agentpy
# %pip install seaborn

import numpy as np
import time
import seaborn as sns
from matplotlib import pyplot as plt
from IPython.display import HTML
import agentpy as ap
from matplotlib.animation import PillowWriter
import socket
import json
import math

CAR_AMOUNT_MULT = 2

def fillGrid(rows, cols, arm_len=11, half_width=7):
    grid = np.ones((rows, cols, 3), dtype=np.uint8) * 255  # white background

    black  = [0, 0, 0]
    green  = [0, 255, 0]
    red    = [255, 0, 0]
    blue   = [0, 0, 255]
    orange = [255, 165, 0]

    mid_r = rows // 2
    mid_c = cols // 2
    border_rows = [mid_r - half_width, mid_r + half_width]
    border_cols = [mid_c - half_width, mid_c + half_width]

    # Left & Right
    for r in border_rows:
        grid[r, 0:arm_len] = black
        grid[r, cols - arm_len:cols] = black

    grid[0:arm_len, mid_c] = black
    grid[rows-arm_len:rows, mid_c] = black

    # Top & Bottom
    for c in border_cols:
        grid[0:arm_len, c] = black
        grid[rows - arm_len:rows, c] = black

    grid[mid_r, 0:arm_len] = black
    grid[mid_r, cols-arm_len:cols] = black

    # Example traffic lights
    grid[mid_c-half_width-1, arm_len-2] = green  # Top Left
    grid[mid_c+half_width+1, arm_len-2] = green  # Bottom Left
    grid[arm_len-2, mid_r+half_width+1] = green  # Top Right
    grid[rows-arm_len+1, mid_r+half_width+1] = green  # Bottom Right


    # #Blue: Origin, Red: Destination, Orange: Stop Point for traffic Light.
    # # Top Sur
    # grid[0, mid_c - half_width//2] = blue 
    # grid[arm_len, mid_c - half_width//2] = orange 
    # grid[0, mid_c + half_width//2] = red

    # # Bottom Norte
    # grid[rows-1, mid_c - half_width//2] = red
    # grid[rows-arm_len-1, mid_c + half_width//2] = orange
    # grid[rows-1, mid_c + half_width//2] = blue

    # # Left Este
    # grid[mid_r - half_width//2, 0] = red
    # grid[mid_r + half_width//2, arm_len] = orange
    # grid[mid_r + half_width//2, 0] = blue

    # # Right Oeste
    # grid[mid_r - half_width//2, cols-1] = blue
    # grid[mid_r - half_width//2, cols-arm_len-1] = orange
    # grid[mid_r + half_width//2, cols-1] = red

    return grid


def fillGridData(rows, cols, arm_len=11, half_width=7):
    collisionLocations = []
    lightLocations = {}
    locationToCoords = {}
    
    # Center lines that define the two 'borders' for each approach
    mid_r = rows // 2
    mid_c = cols // 2
    border_rows = [mid_r - half_width, mid_r + half_width]  # for left/right sides
    border_cols = [mid_c - half_width, mid_c + half_width]  # for top/bottom sides

    # Left & Right sides (vertical borders → horizontal segments going inward)
    for r in border_rows:
        collisionLocations.append((r,i) for i in range(arm_len))
        collisionLocations.append((r,i) for i in range(cols - arm_len,cols))
        
        
    collisionLocations.append((i,mid_c) for i in range(arm_len))
    collisionLocations.append((i,mid_c) for i in range(rows-arm_len,rows))
    # Top & Bottom sides (horizontal borders → vertical segments going inward)
    for c in border_cols:
        collisionLocations.append((i,c) for i in range(arm_len))
        collisionLocations.append((i,c) for i in range(rows - arm_len,rows))

    collisionLocations.append((mid_r,i) for i in range(arm_len))
    collisionLocations.append((mid_r,i) for i in range(cols-arm_len,cols))

    lightLocations = {
        1: (rows-arm_len+1, mid_r+half_width+1), #Junco Norte
        3:(mid_c+half_width+1, arm_len-2), #Roel Este 
        5:(mid_c-half_width-1, arm_len-2), #Junco Sur
        7:(arm_len-2, mid_r+half_width+1) #Roel Oeste
    }

    locationToCoords = { #Spawn locations
        1: (rows-1,mid_c + half_width//2),#Junco Norte
        2: (rows-1,mid_c - half_width//2),#Junco Norte
        3: (mid_r + half_width//2, 0),#Roel Este
        4: (mid_r - half_width//2, 0), #Roel Este
        5: (0,mid_c - half_width//2),#Junco Sur
        6: (0,mid_c + half_width//2),#Junco Sur
        7: (mid_r - half_width//2, cols-1), #Roel Oeste
        8: (mid_r + half_width//2, cols-1)#Roel Oeste
    }

    lightCheckpoints = {
        1: (rows-arm_len-1, mid_c + half_width//2), #Junco Norte
        3: (mid_r + half_width//2, arm_len),#Roel Este
        5: (arm_len, mid_c - half_width//2), #Junco Sur
        7: (mid_r - half_width//2, cols-arm_len-1), #Roel Oeste
    }

    return collisionLocations, lightLocations, locationToCoords,lightCheckpoints

import numpy as np
import matplotlib.pyplot as plt

def my_plot(model, ax, arm_len=11, half_width=7):
   
    rows, cols = model.environment.shape  # expected (35, 35)

    colorDict = {
        'green' : [0,255,0],
        'red' : [255,0,0],
        'yellow' : [255, 255, 0]  
    }
    
    grid = fillGrid(rows,cols)

    for agent, pos in model.environment.positions.items():
        if agent.__class__.__name__ == "TrafficLight":
            grid[pos] = colorDict[agent.get_color()] 
        elif agent.__class__.__name__ == "Vehicle":
            if agent.isCar:
                grid[pos] = [106,13,173]
            else:
                grid[pos] = [255, 255, 0]


    ax.imshow(grid)
    ax.set_xticks([])
    ax.set_yticks([])

# class DummyEnv:
#     shape = (35, 35)

# class DummyModel:
#     environment = DummyEnv()

# fig, ax = plt.subplots(figsize=(5, 5))
# my_plot(DummyModel(), ax) 
# plt.show()


#Car Probabilities

morning_probs = {
    1: 0.9427, #Junco Norte 
    3: 0.9810, #Roel Este 
    5: 0.9941, #Junco Sur 
    7: 0.9731, #Roel Oeste 
}

afternoon_probs = {
    1: 0.9761, #Junco Norte 
    3: 0.9802, #Roel Este 
    5: 0.9936, #Junco Sur 
    7: 0.9753, #Roel Oeste 
}

night_probs = {
    1: 0.9921, #Junco Norte 
    3: 0.9977, #Roel Este 
    5: 0.9884, #Junco Sur 
    7: 0.9680, #Roel Oeste 
}



#Car Amounts

morning_cars = {
    (1, 4): 3, #Junco Norte Izquierda
    (1, 6): 3, #Junco Norte Adelante
    (1, 8): 1, #Junco Norte Derecha
    (3, 6): 1, #Roel Este Izquierda
    (3, 8): 1, #Roel Este Adelante
    (3, 2): 3, #Roel Este Derecha
    (5, 8): 1, #Junco Sur Izquierda
    (5, 2): 1, #Junco Sur Adelante
    (5, 4): 1, #Junco Sur Derecha
    (7, 2): 3, #Roel Oeste Izquierda
    (7, 4): 7, #Roel Oeste Adelante
    (7, 6): 5 #Roel Oeste Derecha
}

afternoon_cars = {
    (1, 4): 3, #Junco Norte Izquierda
    (1, 6): 3, #Junco Norte Adelante
    (1, 8): 1, #Junco Norte Derecha
    (3, 6): 3, #Roel Este Izquierda
    (3, 8): 3, #Roel Este Adelante
    (3, 2): 1, #Roel Este Derecha
    (5, 8): 1, #Junco Sur Izquierda
    (5, 2): 3, #Junco Sur Adelante
    (5, 4): 1, #Junco Sur Derecha
    (7, 2): 3, #Roel Oeste Izquierda
    (7, 4): 5, #Roel Oeste Adelante
    (7, 6): 5 #Roel Oeste Derecha
}

night_cars = {
    (1, 4): 3, #Junco Norte Izquierda
    (1, 6): 3, #Junco Norte Adelante
    (1, 8): 1, #Junco Norte Derecha
    (3, 6): 1, #Roel Este Izquierda
    (3, 8): 1, #Roel Este Adelante
    (3, 2): 3, #Roel Este Derecha
    (5, 8): 1, #Junco Sur Izquierda
    (5, 2): 3, #Junco Sur Adelante
    (5, 4): 1, #Junco Sur Derecha
    (7, 2): 5, #Roel Oeste Izquierda
    (7, 4): 5, #Roel Oeste Adelante
    (7, 6): 3 #Roel Oeste Derecha
}

route_directions = {
    (1, 4): "left", #Junco Norte Izquierda
    (1, 6): "forward", #Junco Norte Adelante
    (1, 8): "right", #Junco Norte Derecha
    (3, 6): "left", #Roel Este Izquierda
    (3, 8): "forward", #Roel Este Adelante
    (3, 2): "right", #Roel Este Derecha
    (5, 8): "left", #Junco Sur Izquierda
    (5, 2): "forward", #Junco Sur Adelante
    (5, 4): "right", #Junco Sur Derecha
    (7, 2): "left", #Roel Oeste Izquierda
    (7, 4): "forward", #Roel Oeste Adelante
    (7, 6): "right" #Roel Oeste Derecha
}

route_lanes = {
    (1, 4): ["left"], #Junco Norte Izquierda
    (1, 6): ["middle"], #Junco Norte Adelante
    (1, 8): ["right"], #Junco Norte Derecha
    (3, 6): ["left"], #Roel Este Izquierda
    (3, 8): ["left"], #Roel Este Adelante
    (3, 2): ["right"], #Roel Este Derecha
    (5, 8): ["left"], #Junco Sur Izquierda
    (5, 2): ["left"], #Junco Sur Adelante
    (5, 4): ["right"], #Junco Sur Derecha
    (7, 2): ["left"], #Roel Oeste Izquierda
    (7, 4): ["left"], #Roel Oeste Adelante
    (7, 6): ["right"], #Roel Oeste Derecha
}


# Locations
locations = {
    1: {"name": "Junco de la Vega Norte Origen", "location": (0, 0), "type": "origin"},
    2: {"name": "Junco de la Vega Norte Destino", "location": (0, 0), "type": "destination"},
    3: {"name": "Garcia Roel Este Origen",       "location": (0, 0), "type": "origin"},
    4: {"name": "Garcia Roel Este Destino",      "location": (0, 0), "type": "destination"},
    5: {"name": "Junco de la Vega Sur Origen",   "location": (0, 0), "type": "origin"},
    6: {"name": "Junco de la Vega Sur Destino",  "location": (0, 0), "type": "destination"},
    7: {"name": "Garcia Roel Oeste Origen",      "location": (0, 0), "type": "origin"},
    8: {"name": "Garcia Roel Oeste Destino",     "location": (0, 0), "type": "destination"}
}

#Traffic Light Phases

phaseDuration = {
    1: 20,
    3: 19,
    5: 19,
    7: 22,
    9: 10, #All Traffic Lights are on red for pedestrians
}

# afternoon_phase = {
#     1: 20,
#     3: 19,
#     5: 19,
#     7: 22,
#     9: 10, #All Traffic Lights are on red for pedestrians
# }

# night_phase = {
#     1: 20,
#     3: 19,
#     5: 19,
#     7: 22,
#     9: 10, #All Traffic Lights are on red for pedestrians
# }

# phaseDuration = {
#     "morning" : morning_phase,
#     "afternoon": afternoon_phase,
#     "night" : night_phase
# }


def generate_flows(time: str, vehicle_pattern, car_probs):
    
    flows = {}
    for origin in range(1, 9, 2):  # origins 1-4
        for destination in range(2, 9, 2):
            if origin + 1 == destination:
                continue  # skip same-node trips
                
            vehicles = vehicle_pattern[(origin, destination)]
            car_prob = car_probs[origin]

            flows[(origin, destination)] = {
                "time": time,
                "vehicles": vehicles,
                "carProbability": car_prob
            }
    return flows


# Generate datasets
morningAmount   = generate_flows("morning", morning_cars, morning_probs)
afternoonAmount = generate_flows("afternoon", afternoon_cars, afternoon_probs)  # default pattern
nightAmount     = generate_flows("night", night_cars, night_probs)      # default pattern

# Group everything
carAmounts = {
    "morning": morningAmount, #2-10
    "afternoon": afternoonAmount, #10-18
    "night": nightAmount #18-2
}

carAmounts

for key in carAmounts.keys():
    for keyTup in carAmounts[key]:
        carAmounts[key][keyTup]["vehicles"] = math.floor(carAmounts[key][keyTup]["vehicles"] * CAR_AMOUNT_MULT)


class TrafficControler(ap.Agent):

    def setup(self, env, trafficLightList, phaseDuration, totalCycle=80, yellowSec=5, minGreen=5, alpha=0.2):
        self.env = env
        self.trafficLightList = trafficLightList
        self.phaseDuration = phaseDuration
        self.time = 0
        self.eventIdx = 0
        self.currentPhase = 1
        self.pedestrianPhase = 9
        self.currentSeconds = 0


        # Control params
        self.totalCycle = totalCycle  
        self.yellowSec = yellowSec       
        self.minGreen = minGreen
        self.alpha = alpha

        self.phaseDuration = {
            id: totalCycle / len(trafficLightList) 
            for id in self.phaseDuration.keys()
            if id != self.pedestrianPhase
        }
        self.phaseDuration[self.pedestrianPhase] = 10

    def update_traffic_lights(self, cars_waiting):
        
        """
        cars_waiting: dict {id : number of cars waiting at each light}
        self.green_times: dict {id : green time from previous cycle}
        """
        #Remove pedestrian phase
        green_times = {id : value for id,value in self.phaseDuration.items() if id != self.pedestrianPhase}  
        
        for id,_ in self.phaseDuration.items():
            if(id not in cars_waiting and id != self.pedestrianPhase):
                cars_waiting[id] = 0

        print(cars_waiting)

        #No cars waiting = no changes
        total_cars = sum(cars_waiting.values())
        if total_cars == 0:
            return
        
        # Ideal proportional allocation
        new_green_times = {
            lid: (cars / total_cars) * self.totalCycle 
            for lid, cars in cars_waiting.items()
        }
        
        # Ensure that all times are the minimum green time
        new_green_times = {
            lid: max(self.minGreen, g) 
            for lid, g in new_green_times.items()
        }
        
        # Normalize by creating a scale to ensure sum of 80
        scale = self.totalCycle / sum(new_green_times.values())
        new_green_times = {lid: g * scale for lid, g in new_green_times.items()}
        
        # Smoothing Function to avoid drastic changes
        smoothed_green_times = {
            lid: green_times.get(lid, 0) + self.alpha * (g_new - green_times.get(lid, 0))
            for lid, g_new in new_green_times.items()
        }
        
        # Normalize smoothing function
        scale = self.totalCycle / sum(smoothed_green_times.values())
        smoothed_green_times = {lid: g * scale for lid, g in smoothed_green_times.items()}
        
        #Floor values
        smoothed_green_times = {lid: g//1 for lid, g in smoothed_green_times.items()} #Floor values
        
        #Add pedestrian phase
        smoothed_green_times[self.pedestrianPhase] = 10 + 80 - sum(smoothed_green_times.values())
        # Update controller state
        self.phaseDuration = smoothed_green_times

        print(self.phaseDuration)


    def execute(self):
        print("Seconds Controller:",self.currentSeconds)
        self.model.stepLights = []
        self.currentSeconds += self.model.secondsInStep 
        if(self.currentSeconds >= self.phaseDuration[self.currentPhase]):
            self.currentSeconds = self.currentSeconds % self.phaseDuration[self.currentPhase]

            if(self.currentPhase != self.pedestrianPhase):
                self.trafficLightList[self.currentPhase].set_color('red')
                self.model.stepLights.append({'light':self.currentPhase,'color':'red'})
            self.currentPhase = (1 if self.currentPhase == self.pedestrianPhase else self.currentPhase + 2)
            if(self.currentPhase != self.pedestrianPhase):
                self.trafficLightList[self.currentPhase].set_color('green')
                self.model.stepLights.append({'light':self.currentPhase,'color':'green'})
        
        if(self.currentPhase != self.pedestrianPhase and 
           (self.currentSeconds >= self.phaseDuration[self.currentPhase] - self.yellowSec)):
            self.trafficLightList[self.currentPhase].set_color('yellow')
            self.model.stepLights.append({'light':self.currentPhase,'color':'yellow'})

        


class TrafficLight(ap.Agent):

    def setup(self, env, location):
        self.env = env
        self.color = 'red' #green,yellow,red
        self.location = location
        
    def get_color(self):
        return self.color
    
    def set_color(self, color):
        self.color = color
    
    def get_location(self):
        return self.location
        

    def execute(self):
        1+1
        #print(f'Execute trafficlight at: {self.location} on color: {self.color}')


class Vehicle(ap.Agent):

    #def setup(self, env, isCar, inRightLane, direction, destination):
    def setup(self, env, trafficLight, startPos, lightCheckpoint, destination, isCar):
        self.env = env
        self.isCar = isCar
        self.trafficLight = trafficLight
        self.lightCheckpoint = lightCheckpoint
        self.destination = destination
        rowPos,colPos = startPos
        self.isMoving = False
        self.passedLight = False
        self.direction = ()
        self.rows, self.cols = self.env.shape
        self.toRemove = False
        self.directionChange = False
        self.movingUnits = 1
        dest_x, dest_y = self.destination
        if(rowPos == 0): #Top
            self.direction = (self.movingUnits,0)
        elif(colPos == 0): #Left
            self.direction = (0,self.movingUnits)
        elif(rowPos == 34): #Bottom NEEDS CHANGE
            self.direction = (-self.movingUnits,0)
        else:
            self.direction = (0,-self.movingUnits)

        if(dest_x == rowPos or dest_y ==colPos):
            self.directionChange = True
            

    def execute(self):
        x,y = np.array(self.direction) + self.env.positions[self]
        if(self.env.positions[self] == self.lightCheckpoint): #Has passed, can continue moving
            self.passedLight = True
        if not (0 <= x < self.rows and 0 <= y < self.cols): #Is in border, should be removed
            self.toRemove = True
        elif(self.passedLight or self.trafficLight.get_color() != 'red'): #moves if passed light or light is not red
            self.env.move_by(self, self.direction)
        elif (not self.passedLight and (x,y) != self.lightCheckpoint and (x,y) not in self.env.positions.values()): #moves if doesn't have another car in front
            self.env.move_by(self, self.direction)

        if not self.directionChange:
            dest_x, dest_y = self.destination
            # Coming from vertical 
            if (self.direction in [(self.movingUnits,0), (-self.movingUnits,0)] and x == dest_x):
                # turn horizontal
                if dest_y > y:  # destination is to the right
                    self.direction = (0, self.movingUnits)
                else: # destination to the left
                    self.direction = (0, -self.movingUnits)
                self.directionChange = True

            # Coming from horizontal 
            elif (self.direction in [(0,self.movingUnits), (0,-self.movingUnits)] and y == dest_y):
                # turn vertical
                if (dest_x > x):  # destination below
                    self.direction = (self.movingUnits, 0)
                else:  # destination above
                    self.direction = (-self.movingUnits, 0)
                self.directionChange = True

        


class TrafficModel(ap.Model):

    def setup(self):
        self.carAmounts = self.p.carAmounts
        self.route_lanes = self.p.route_lanes
        self.route_directions = self.p.route_directions
        self.locations = self.p.locations
        self.phaseDuration = self.p.phaseDuration
        self.lightLocations = self.p.lightLocations
        self.collisionLocations = self.p.collisionLocations
        self.locationToCoords = self.p.locationToCoords
        self.lightCheckpoints = self.p.lightCheckpoints
        self.currentHour = 0
        self.environment = ap.Grid(self, (35, 35))
        self.timeInSeconds = 0
        self.cycleDuration = 90
        self.secondsInStep = 2
        self.currentPeriod = "morning" #morning, afternoon or night
        self.trafficLightDict = {}
        self.stepLightsRef = []
        self.agentsQueue = {i:[] for i in self.lightLocations.keys()}
        self.stepLights = []
        self.stepCars = []
        self.cars_waiting = {}
        self.referenceToWaitingCars = self.p.referenceToWaitingCars
        
        for i in self.lightLocations.keys():
            currentTrafficLignt = TrafficLight(self,self.environment, self.lightLocations[i])
            self.trafficLightDict[i] = currentTrafficLignt
            self.environment.add_agents([currentTrafficLignt], positions=[self.lightLocations[i]])


        self.trafficControlerAgent = TrafficControler(self, self.environment, self.trafficLightDict, self.phaseDuration)
        self.environment.add_agents([self.trafficControlerAgent], positions=[(0,0)])

        self.trafficLightDict[1].set_color('green')
        
        stepJSON = {'stepCars':self.stepCars,'stepLights':[{'light':1,'color':'green'}]}
        self.p.s.sendall(json.dumps(stepJSON).encode("utf-8"))

        for lightKey in self.trafficLightDict.keys():
            #if(not self.agentsQueue[lightKey]):
            for origin,destination in self.carAmounts[self.currentPeriod].keys():
                if(origin == lightKey):
                    for i in range(self.carAmounts[self.currentPeriod][(origin,destination)]["vehicles"]):
                        lanes = self.route_lanes[(origin,destination)]
                        isCar = np.random.random() < self.carAmounts[self.currentPeriod][(origin,destination)]["carProbability"] 
                        self.agentsQueue[lightKey].append(([Vehicle(self,self.environment, self.trafficLightDict[lightKey],self.locationToCoords[lightKey],
                            self.lightCheckpoints[lightKey],self.locationToCoords[destination],isCar)],[self.locationToCoords[lightKey]], 
                            {
                                'origin':origin, 'destination':destination, 'direction': self.route_directions[(origin,destination)], 
                                'lane': lanes[0] if len(lanes) == 1 else str(np.random.choice(lanes)),
                                'isCar': isCar
                            }))
        time.sleep(self.secondsInStep)

    def step(self):
        stepJSON = {}
        self.stepCars = []
        print("Time: ", self.timeInSeconds )

        if(2 <= self.currentHour < 10):
            self.currentPeriod = "morning"
        elif (10 <= self.currentHour < 18):
            self.currentPeriod = "afternoon"
        else:
            self.currentPeriod = "night"

        
        if(self.timeInSeconds == 80): #Fill the queues at the pedestrian phase.
            for lightKey in self.trafficLightDict.keys():
                #if(not self.agentsQueue[lightKey]):
                for origin,destination in self.carAmounts[self.currentPeriod].keys():
                    if(origin == lightKey):
                        for i in range(self.carAmounts[self.currentPeriod][(origin,destination)]["vehicles"]):
                            lanes = self.route_lanes[(origin,destination)]
                            isCar = np.random.random() < self.carAmounts[self.currentPeriod][(origin,destination)]["carProbability"] 
                            self.agentsQueue[lightKey].append(([Vehicle(self,self.environment, self.trafficLightDict[lightKey],self.locationToCoords[lightKey],
                                self.lightCheckpoints[lightKey],self.locationToCoords[destination],isCar)],[self.locationToCoords[lightKey]], 
                                {
                                    'origin':origin, 'destination':destination, 'direction': self.route_directions[(origin,destination)], 
                                    'lane': lanes[0] if len(lanes) == 1 else str(np.random.choice(lanes)),
                                    'isCar': isCar
                                }))
                            
            # # Collect waiting cars
            # self.cars_waiting = {
            #     phase : len(self.agentsQueue.get(phase, []))
            #     for phase in self.trafficLightDict.keys()
            #     if phase != self.trafficControlerAgent.pedestrianPhase
            # }
            
            # #Get cars that didn't cross
            # carsWaiting = 0
            # for agent, pos in self.environment.positions.items():
            #     if agent.__class__.__name__ == "Vehicle" and not agent.passedLight:
            #         carsWaiting += 1

            
            # print(carsWaiting ,end=', ')
                                

        #Spawn cars if its the light is on red
        for lightKey in self.trafficLightDict.keys():
            if(self.agentsQueue[lightKey]):
                currentAgent, position, extra = self.agentsQueue[lightKey][-1]
                if(position[0]): #Check if there is no car at the origin
                    #self.environment.add_agents(currentAgent, positions=position)
                    self.agentsQueue[lightKey].pop()
                    self.stepCars.append(extra)
            

        stepJSON = {'stepCars':self.stepCars,'stepLights':self.stepLights}
        #print ("Received from server: ",self.p.from_server.decode("ascii"))
        self.p.s.sendall(json.dumps(stepJSON).encode("utf-8"))

        to_remove = [agent for agent in self.environment.agents if agent.__class__.__name__ == "Vehicle" and agent.toRemove]
        self.environment.remove_agents(to_remove)
        #Update Lights
        if((self.timeInSeconds + self.secondsInStep) % self.cycleDuration == 0):
            self.currentHour =  (self.currentHour + 1) % 24
        self.environment.agents.execute()
        
        #Comment this section to disable smart traffic light time change
        if(self.timeInSeconds == self.cycleDuration - self.secondsInStep):
            # Update phase durations adaptively
            self.p.s.sendall(('{"command": "countCars"}').encode("utf-8"))
            json_str = s.recv(4096).decode("utf-8")
            cars_waiting = {item["id"]: item["count"] for item in json.loads(json_str)["items"]}
            print(cars_waiting)
            referenceToWaitingCars.append(sum(cars_waiting.values())) 
            self.trafficControlerAgent.update_traffic_lights(cars_waiting) 

        self.timeInSeconds = (self.timeInSeconds + self.secondsInStep) % self.cycleDuration
        time.sleep(self.secondsInStep)
        


s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("127.0.0.1", 1101))



fig, ax = plt.subplots()
referenceToWaitingCars = []
collisionLocations, lightLocations, locationToCoords, lightCheckpoints = fillGridData(35,35)
parameters = {
    'steps': 90*12,'carAmounts': carAmounts, 'locations':locations, 'phaseDuration': phaseDuration, 
    'lightLocations':lightLocations, 'collisionLocations' : collisionLocations, 'locationToCoords': locationToCoords,
    'lightCheckpoints' : lightCheckpoints, 'route_directions': route_directions, 'route_lanes' : route_lanes,
    's':s, 'referenceToWaitingCars': referenceToWaitingCars
} 
trafficModel = TrafficModel(parameters)
results = trafficModel.run()

print("Total Cars Waiting")
for item in referenceToWaitingCars:
    print(item, ",", sep="", end="")



#animation = ap.animate(trafficModel, fig, ax, my_plot)

#../.venv/bin/python3 TrafficSimulationScript.py 