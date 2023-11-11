import random
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule

class StreetAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

class SidewalkAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

class ConeAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

class CarAgent(Agent):
    def __init__(self, unique_id, model, failure_prob, speed):
        super().__init__(unique_id, model)
        self.failure_prob = failure_prob
        self.color = "blue"
        self.speed = speed  # Número de pasos antes de moverse
        self.steps_to_move = speed  # Inicializamos la variable de pasos a la velocidad
        self.max_speed = 1
        self.acceleration = 1
        self.collision = False
        

    def see_if_empty(self):
        x, y = self.pos
        
        width = self.model.grid.height
        search_cells = width - y -1
        visible_cells = []

        print(search_cells)
        if search_cells == 1:
            return False
        elif search_cells > 3:
            search_cells = 3

        
        for i in range(1, search_cells+1):
            visible_cells.append((x, y+i))
            
        

        for cell in visible_cells:
            cell_content = self.model.grid.get_cell_list_contents([cell])
            for agent in cell_content:
                if not isinstance(agent, StreetAgent):
                    return False  

        return True 
    
    def see_if_empty_sides(self):
        x, y = self.pos
        max_x, max_y = self.model.grid.width, self.model.grid.height

        visible_cells = [
            (x+1, y+1) if x+1 < max_x and y+1 < max_y else None, 
            (x-1, y+1) if x-1 >= 0 and y+1 < max_y else None,  
        ]

        empty_cells = []

        for cell in visible_cells:
            if cell is not None:
                cell_content = self.model.grid.get_cell_list_contents([cell])

                if all(isinstance(agent, StreetAgent) for agent in cell_content):
                    empty_cells.append(cell)

        return empty_cells 
        
    def step(self):
        x, y = self.pos
        #possible_steps = self.see()

        my_cell = self.model.grid.get_cell_list_contents([self.pos])

        
        
        if(len(my_cell)>2):
            self.color = "yellow"
            if self.collision == False:
                self.model.collision_agent +=1
                self.collision = True
            return
    
        else:
                
           
            
            if self.steps_to_move == 0:
                next_x = x
                next_y = y + 1
                if 0 <= next_x < self.model.grid.width and 0 <= next_y < self.model.grid.height:  # Comprueba si la celda está dentro de los límites de la cuadrícula
                    next_cell = self.model.grid.get_cell_list_contents([(next_x, next_y)])
                    next_cell_is_free = all(isinstance(agent, StreetAgent) for agent in next_cell)
                else:
                    next_cell_is_free = False 
                    self.model.grid.move_agent(self, (self.pos[0], self.pos[1]+1))

                if next_cell_is_free:
                    # Escoger una celda delante del agente al azar dependiendo de la probabilidad
                    if random.random() > self.failure_prob:
                        possible_directions = [0, 1, -1]  
                        direction = random.choice(possible_directions)
                        new_x = x + direction
                        new_y = y + 1

                        
                        self.model.grid.move_agent(self, (new_x, new_y))
                        self.steps_to_move = self.speed
                        

                    else:
                        self.model.grid.move_agent(self, (next_x, next_y))
                        self.steps_to_move = self.speed
                else:
                    empty_sides = self.see_if_empty_sides()
                    if empty_sides:
                        # Cambiar de carril al primer carril vacío disponible
                        self.model.grid.move_agent(self, empty_sides[0])
                        # Reiniciar los pasos para moverse
                        self.steps_to_move = max(1, self.max_speed - self.speed + 1)


            else:
                self.steps_to_move -=1


class CarAgent2(Agent):
    def __init__(self, unique_id, model, failure_prob, speed):
        super().__init__(unique_id, model)
        self.failure_prob = failure_prob
        self.color = "green"
        self.speed = speed  # Número de pasos antes de moverse
        self.steps_to_move = speed  # Inicializamos la variable de pasos a la velocidad
        self.max_speed = 1
        self.acceleration = 2
        self.collision = False
        

    def see_if_empty(self):
        x, y = self.pos
        
        width = self.model.grid.height
        search_cells = width - y -1
        visible_cells = []

        print(search_cells)
        if search_cells == 1:
            return False
        elif search_cells > 3:
            search_cells = 3

        
        for i in range(1, search_cells+1):
            visible_cells.append((x, y+i))
            
        

        for cell in visible_cells:
            cell_content = self.model.grid.get_cell_list_contents([cell])
            for agent in cell_content:
                if not isinstance(agent, StreetAgent):
                    return False 

        return True  
    
    def see_if_empty_sides(self):
        x, y = self.pos
        max_x, max_y = self.model.grid.width, self.model.grid.height

        visible_cells = [
            (x+1, y+1) if x+1 < max_x and y+1 < max_y else None,  # Arriba
            (x-1, y+1) if x-1 >= 0 and y+1 < max_y else None,  # Arriba 
        ]

        empty_cells = []

        for cell in visible_cells:
            if cell is not None:
                cell_content = self.model.grid.get_cell_list_contents([cell])

                if all(isinstance(agent, StreetAgent) for agent in cell_content):
                    empty_cells.append(cell)

        return empty_cells
 
        
    def step(self):
        # Verificar si el agente puede moverse

        print(f"Steps to move: {str(self.steps_to_move)}")
    # Verificar si el agente puede moverse

        my_cell = self.model.grid.get_cell_list_contents([self.pos])

        
        
        if(len(my_cell)>2):
            self.color = "yellow"
            if self.collision == False:
                self.model.collision_agent2 +=1
                cuadrant = self.model.height // 4
                pos_cuadrant = max(1, (self.pos[1] - 1) // cuadrant + 1)

                
                
                self.model.cuadrant[pos_cuadrant-1] += 1
                self.collision = True

            
            return
        else:
            if self.steps_to_move <= 0:
                # Verificar si las celdas adelante están vacías
                if self.see_if_empty():
                    # Si las celdas adelante están vacías, el agente puede aumentar su velocidad
                    self.speed = max(self.speed - self.acceleration, self.max_speed)
                    # Mover al agente
                    self.model.grid.move_agent(self, (self.pos[0], self.pos[1] + 1))
                    # Reiniciar los pasos para moverse
                    self.steps_to_move = self.speed
                else:
                    # Si las celdas adelante no están vacías, el agente comienza a desacelerar
                    self.speed = self.speed + self.acceleration
                    # Verificar si hay celdas vacías a los lados para cambiar de carril
                    empty_sides = self.see_if_empty_sides()
                    if empty_sides:
                        # Cambiar de carril al primer carril vacío disponible
                        self.model.grid.move_agent(self, empty_sides[0])
                        # Reiniciar los pasos para moverse
                        self.steps_to_move = max(1, self.max_speed - self.speed + 1)
            
    
            self.steps_to_move -=1

        print(f"Soy el carro {str(self.unique_id)}, mi velocidad es {str(self.speed)}")


class TrafficModel(Model):
    def __init__(self, width, height, num_cars, street_width):
        street_width = street_width-1
        super().__init__()
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.collision_agent = 0
        self.collision_agent2 = 0
        self.cuadrant = [0, 0, 0, 0]
        self.height = height

        
        model_reporters = {"AverageSpeed": self.compute_average_speed, 
                           "CollisionsAgent2": self.compute_collision2, 
                           "CollisionsAgent": self.compute_collision,
                           "Cuadrant": self.compute_cuadrant}
        
        self.datacollector = DataCollector(model_reporters=model_reporters)

        center_x = self.grid.width // 2  # Posición x del centro
        center_y = self.grid.height // 2  # Posición y del centro
        print(center_x)
        print(center_y)

        # Asegurarse de que el ancho de la calle sea par
        if street_width % 2 == 0:
            left_width = street_width // 2
            right_width = street_width // 2
        else:
            left_width = street_width // 2
            right_width = street_width // 2 + 1

        

        for x in range(center_x - width // 2, center_x + width // 2 + 1):
            for y in range(center_y - height // 2, center_y + height // 2 + 1):
                if 0 <= x < self.grid.width and 0 <= y < self.grid.height:
                    if (
                        center_x - left_width <= x <= center_x + right_width
                        or center_y - left_width <= y <= center_y + right_width
                    ):
                        street_agent = StreetAgent((x, y), self)
                        self.grid.place_agent(street_agent, (x, y))
                    else:
                        sidewalk_agent = SidewalkAgent((x, y), self)
                        self.grid.place_agent(sidewalk_agent, (x, y))

    
        cone_agent = ConeAgent(10000, self)
        self.grid.place_agent(cone_agent, (5,10))
        cone_agent = ConeAgent(10001, self)
        self.grid.place_agent(cone_agent, (9,7))
        cone_agent = ConeAgent(10002, self)
        self.grid.place_agent(cone_agent, (10,3))
        cone_agent = ConeAgent(10003, self)
        self.grid.place_agent(cone_agent, (6,14))
        cone_agent = ConeAgent(10004, self)
        self.grid.place_agent(cone_agent, (11,9))
        cone_agent = ConeAgent(10005, self)
        self.grid.place_agent(cone_agent, (2,2))
        cone_agent = ConeAgent(10004, self)
        self.grid.place_agent(cone_agent, (15,9))
        cone_agent = ConeAgent(10005, self)
        self.grid.place_agent(cone_agent, (13,2))
        cone_agent = ConeAgent(10005, self)
        self.grid.place_agent(cone_agent, (13,10))


        car = CarAgent(1, self, 0.2, 10)
        self.grid.place_agent(car, (4, 0))
        self.schedule.add(car)

        car = CarAgent2(2, self, 0.2, 10)
        self.grid.place_agent(car, (5, 0))
        self.schedule.add(car)

        car = CarAgent(3, self, 0.2, 10)
        self.grid.place_agent(car, (6, 0))
        self.schedule.add(car)

        car = CarAgent2(4, self, 0.2, 10)
        self.grid.place_agent(car, (7, 0))
        self.schedule.add(car)

        car = CarAgent(5, self, 0.2, 10)
        self.grid.place_agent(car, (8, 0))
        self.schedule.add(car)

        car = CarAgent2(6, self, 0.2, 10)
        self.grid.place_agent(car, (9, 0))
        self.schedule.add(car)

        car = CarAgent(7, self, 1, 11)
        self.grid.place_agent(car, (10, 0))
        self.schedule.add(car)

        car = CarAgent2(8, self, 1, 10)
        self.grid.place_agent(car, (11, 0))
        self.schedule.add(car)

        car = CarAgent(9, self, 1, 10)
        self.grid.place_agent(car, (12, 0))
        self.schedule.add(car)

        car = CarAgent2(10, self, 1, 10)
        self.grid.place_agent(car, (13, 0))
        self.schedule.add(car)

        car = CarAgent(11, self, 1, 10)
        self.grid.place_agent(car, (14, 0))
        self.schedule.add(car)

        car = CarAgent2(12, self, 1, 10)
        self.grid.place_agent(car, (15, 0))
        self.schedule.add(car)

    def compute_collision2(self):
        return self.collision_agent2
    
    def compute_collision(self):
        return self.collision_agent
    def compute_cuadrant(self):
        return self.cuadrant
    
    def compute_average_speed(self):
        # Calcula la velocidad promedio de todos los agentes CarAgent2
        car_agents = [agent for agent in self.schedule.agents if isinstance(agent, CarAgent2)]
        average_speed = sum(agent.speed for agent in car_agents) / len(car_agents)
        return average_speed

    def step(self):
        
        self.datacollector.collect(self)  
        self.schedule.step()
        
        

def agent_portrayal(agent):
    if isinstance(agent, StreetAgent):
        portrayal = {"Shape": "rect", "Color": "grey", "Filled": "true", "Layer": 0, "w": 1, "h": 1}
    elif isinstance(agent, SidewalkAgent):
        portrayal = {"Shape": "rect", "Color": "lightgrey", "Filled": "true", "Layer": 0, "w": 1, "h": 1}
    elif isinstance(agent, CarAgent):
        portrayal = {"Shape": "circle", "Color": agent.color, "Filled": "true", "Layer": 1, "r": 0.5}
    elif isinstance(agent, CarAgent2):
        portrayal = {"Shape": "circle", "Color": agent.color, "Filled": "true", "Layer": 1, "r": 0.5}
    elif isinstance(agent, ConeAgent):
        portrayal = {"Shape": "circle", "Color": "orange", "Filled": "true", "Layer": 1, "r": 0.5}
    else:
        portrayal = None
    return portrayal

width = 20
height = 20
num_cars = 1

grid = CanvasGrid(agent_portrayal, width, height, 500, 500)

"""""
# Crea una instancia del servidor de visualización y define los elementos a mostrar
server = ModularServer(
    TrafficModel,
    [grid],
    "Modelo de Tráfico",
    {"width": width, "height": height, "num_cars": num_cars, "street_width": 14  }
)

# Inicia el servidor de visualización
server.port = 8521  # Elige un puerto para la visualización
server.launch()

"""
import matplotlib.pyplot as plt
import numpy as np
# Asumiendo que 'model' es una instancia de tu modelo TrafficModel
model = TrafficModel(width, height, num_cars, 14)
# Número de veces que quieres ejecutar el modelo
num_runs = 100

# Almacena los datos de todas las ejecuciones
all_collisions_agent = []
all_collisions_agent2 = []
all_cuadrant = []
all_collisions = []
all_avg_speeds = []

for _ in range(num_runs):
    # Asumiendo que 'model' es una instancia de tu modelo TrafficModel
    model = TrafficModel(width, height, num_cars, 20)

    # Ejecuta el modelo
    for i in range(250):  # Ejecuta el modelo durante 300 pasos
        model.step()

    # Recupera los datos recopilados
    model_data = model.datacollector.get_model_vars_dataframe()
    all_collisions_agent.append(model_data['CollisionsAgent'].iloc[-1])
    all_collisions_agent2.append(model_data['CollisionsAgent2'].iloc[-1])
    all_cuadrant.append(model_data["Cuadrant"].iloc[-1])

    all_collisions.append(model_data['CollisionsAgent2'])
    all_avg_speeds.append(model_data['AverageSpeed'])

# Calcula el promedio de los datos de todas las ejecuciones
avg_collisions_agent = sum(all_collisions_agent) / num_runs
avg_collisions_agent2 = sum(all_collisions_agent2) / num_runs
avg_cuadrant = [sum(x)/num_runs for x in zip(*all_cuadrant)]
avg_collisions = sum(all_collisions) / num_runs
avg_avg_speeds = sum(all_avg_speeds) / num_runs



#Grafica de colisiones por step promedio
plt.figure()
# Ahora puedes trazar los datos promedio
fig, ax1 = plt.subplots(figsize=(10, 8))

color = 'tab:red'
ax1.set_xlabel('Tiempo (steps)')
ax1.set_ylabel('Coches que colisionaron', color=color)
ax1.plot(avg_collisions, color=color, label='Promedio de colisiones')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Velocidad media', color=color)  
ax2.plot(avg_avg_speeds, color=color, label='Promedio de velocidad media')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  
plt.title('Promedio de coches colisionados y velocidad media vs Tiempo para varias ejecuciones')
plt.legend()
plt.show()

# Ahora puedes trazar los datos promedio
#Grafica de número de colisiones por cuadrante
plt.figure()
cuadrantes = ['Cuadrante 1', 'Cuadrante 2', 'Cuadrante 3', 'Cuadrante 4']
plt.pie(avg_cuadrant, labels=cuadrantes, autopct='%1.1f%%', startangle=90)
plt.legend(cuadrantes, loc="center left", bbox_to_anchor=(1, 0.5))
plt.title('Porcentaje de Colisiones por Cuadrante')
plt.show()

# Grafica con el número de colisiones del agente 1 y 2
plt.figure()
fig, ax = plt.subplots()
ind = np.arange(1)
width = 0.35
rects1 = ax.bar(ind - width/2, [avg_collisions_agent], width, label='Carros A(azul)')
rects2 = ax.bar(ind + width/2, [avg_collisions_agent2], width, label='Carro B(verde)')
ax.set_xlabel('Agents')
ax.set_ylabel('Collisions')
ax.set_title('Comparación de colisiones de Carros A y Carros B')
ax.legend()
plt.show()
