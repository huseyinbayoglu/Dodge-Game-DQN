import numpy as np
import random
import pygame
import json
from typing import List, Tuple, Dict, Union, Optional
import pandas as pd 


with open('envconfig.json') as config_file:
    config = json.load(config_file)

BLOCK_SIZE = config["BLOCK_SIZE"]
SPEED = config["SPEED"]
# rgb colors
WHITE = tuple(config['COLORS']['WHITE'])
RED = tuple(config['COLORS']['RED'])
BLUE = tuple(config['COLORS']['BLUE'])
BLACK = tuple(config['COLORS']['BLACK'])
YELLOW = tuple(config['COLORS']['YELLOW'])
GREEN = tuple(config['COLORS']['GREEN'])
GRAY = tuple(config['COLORS']['GRAY'])

Number_of_balls = config["NUMBER_OF_BALLS"]

rewards = {
    "nothing":-.1,
    "closer":.7,
    "far":-1,
    "eating_food":5,
    "collision":-7,
    "wall":-3
}

object_map = {
 # [up,down,left,right]
 -1:[0,0,0,0,1,0,0],
 0: [0,0,0,0,0,0,0],
 1: [0,0,0,0,0,1,0],
 2: [0,0,0,0,0,0,1],
 3: [1,0,0,1,0,0,0],
 4: [1,0,1,0,0,0,0],
 5: [0,1,0,1,0,0,0],
 6: [0,1,1,0,0,0,0],
 7: [1,0,0,0,0,0,0],
 8: [0,1,0,0,0,0,0],
 9: [0,0,1,0,0,0,0],
 10: [0,0,0,1,0,0,0]
}

reverse_object_map = {
 # [up,down,left,right]
(0,0,0,0,1,0,0):-1,
(0,0,0,0,0,0,0):0,
(0,0,0,0,0,1,0):1,
(0,0,0,0,0,0,1):2,
(1,0,0,1,0,0,0):3,
(1,0,1,0,0,0,0):4,
(0,1,0,1,0,0,0):5,
(0,1,1,0,0,0,0):6,
(1,0,0,0,0,0,0):7,
(0,1,0,0,0,0,0):8,
(0,0,1,0,0,0,0):9,
(0,0,0,1,0,0,0):10
}

def encode_grid_directions(grid):
    # Grid'deki her pozisyonu eksenlere göre kodla
    encoded_grid = np.array([object_map[val] for val in grid.flatten()])
    return encoded_grid.flatten()

def decode_grid_directions(array:np.ndarray):
    array = array[:175]
    # Her 7 elemanlık dilimi alıp liste içinde tutmak
    seven_chunks = np.array([reverse_object_map[array[i:i+7]] for i in range(0, len(array), 7)]).reshape(5,5)
    return seven_chunks

class Env():
    pass 

class Point(object):
    __slots__ = ('x', 'y')
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
    
    def copy(self) -> 'Point':
        x = self.x
        y = self.y
        return Point(x, y)

    def to_dict(self) -> Dict[str, int]:
        d = {}
        d['x'] = self.x
        d['y'] = self.y
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, int]) -> 'Point':
        return Point(d['x'], d['y'])
    
    def __eq__(self, other: Union['Point', Tuple[int, int]]) -> bool:
        if isinstance(other, tuple) and len(other) == 2:
            return other[0] == self.x and other[1] == self.y
        elif isinstance(other, Point) and self.x == other.x and self.y == other.y:
            return True
        return False

    def __sub__(self, other: Union['Point', Tuple[int, int]]) -> int:
        """ Return the Manhattan distance"""
        if isinstance(other, tuple) and len(other) == 2:
            diff_x = abs(self.x - other[0])
            diff_y = abs(self.y - other[1])
            return diff_x + diff_y     
        elif isinstance(other, Point):
            diff_x = abs(self.x - other.x)
            diff_y = abs(self.y - other.y)
            return diff_x + diff_y
    
    def __str__(self) -> str:
        return '({}, {})'.format(self.x, self.y)

class Ball:
    """
    The balls that agent should avoid. Balls move linearly. If a ball collides 
    with the character then the game is over.
    """
    def __init__(self,
                location:Point,
                vx:int,
                vy:int,
                color:tuple=RED):
        
        self.location = location
        self.vx = vx
        self.vy = vy
        self.color = color

        self.number = self.get_number()
    
    def get_number(self):
        """
        Returns the number that identifies the direction in which the ball is going.

        3 for going to up and right,
        4 for going to up and left,
        5 for going to down and right,
        6 for going to down and left,
        7 for going vertically up,
        8 for going vertically down,
        9 for going horizontally left,
        10 for going horizontally right,

        """
        self.the_data = {"up and right":3,
                    "up and left":4, 
                    "down and right":5, 
                    "down and left":6,
                    "vertically up":7,
                    "vertically down":8,
                    "horizontally left":9,
                    "horizontally right":10} 
    
        if self.vx == 1 and self.vy == -1:
            number = self.the_data["up and right"] 
        elif self.vx == -1 and self.vy == -1:
            number = self.the_data["up and left"]
        elif self.vx == 1 and self.vy == 1:
            number = self.the_data["down and right"]
        elif self.vx == -1 and self.vy == 1:
            number = self.the_data["down and left"]
        elif self.vx == 0 and self.vy == -1:
            number = self.the_data["vertically up"]
        elif self.vx == 0 and self.vy == 1:
            number = self.the_data["vertically down"]
        elif self.vx == -1 and self.vy == 0:
            number = self.the_data["horizontally left"]
        elif self.vx ==1 and self.vy == 0:
            number = self.the_data["horizontally right"]

        # if no moves 
        elif self.vx == 0 and self.vy == 0:
            self.vx = 1
            self.vy = -1
            number = self.the_data["up and right"] 

        return number
    
    def __str__(self) -> str:
        return """
                Location :({},{})\n
                Vx :{} , Vy :{}}\n
                number : {}\n
                color :{}
        """.format(self.location["x"],self.location["y"],self.vx,self.vy,self.number,self.color)

class State(object):
    """ Its a class that represent the game state"""
    def __init__(self,width=9,height=9) -> None:
        self.w = width
        self.h = height 

        # The Game Grid. This represents to whole game.
        self.game_grid = np.zeros((self.h, self.w)) 
        self.agent_grid = np.zeros((5,5))
        self.score = 0


        self.agent_location = self.add_agent()
        self.food_location = self.add_goal()
        self.balls = []
        self.add_balls() 

        self.update_agent_grid()
        
    def __str__(self) -> str:
        text = ""
        list1 = self.game_grid.tolist()
        for row in list1:
            for number in row:
                text += "    " +str(number)
            text += "\n"
        return text

    def save_dict(self) -> Dict:
        dict_state = {}
        for index1,row in enumerate(self.game_grid.tolist()):
            for index2,value in enumerate(row):
                id = "("+str(index2) +","+ str(index1)+")" # As Points not zero indexed and (x,y)
                dict_state[id] = value
        return dict_state
    
    @classmethod
    def load_from_dict(cls,dict):
        pass

    def get_character_location(self) -> Point:
        return self.agent_location 
    
    def get_food_location(self) -> Point:
        return self.food_location 

    def get_balls_location(self) -> list[Point]:
        pass

    def add_agent(self) -> Point:
        """
        Add the character to the game. Returns the character location which is a random location.
        """
        # Choose a random location
        location = Point(random.randint(1,self.w-1),random.randint(1,self.h-1))
    
        
        # Agent is represented by a 1. The middle of the grid
        self.agent_grid[2][2] = 1

        self.game_grid[location.y,location.x] = 1
        
        return location

    def add_goal(self) -> Point:
        """
        Add the goal to the game. Returns the goal location which is a random location.
        """
        # Choose a random location
        location = Point(random.randint(0, self.w - 1),random.randint(0, self.h - 1))

        while location - self.agent_location < 8:
            location = Point(random.randint(0, self.w - 1),random.randint(0, self.h - 1))
        # Get a random location until it is not occupied
        while self.game_grid[location.y][location.x] == 1:
            location = Point(random.randint(0, self.w - 1),random.randint(0, self.h - 1))
        

        
        # Goal is represented by a -1
        self.game_grid[location.y][location.x] = -1


        return location
    
    def add_balls(self,number=Number_of_balls):
        """
        Add the balls to the game.
        """
        for _ in range(number):
            while True:
                x = random.randint(0,self.w-1)
                y = random.randint(0,self.h-1)
                vx,vy = random.choice([-1,1,0]), random.choice([-1,1,0])
                if x != self.agent_location.x or y != self.agent_location.y:
                    break
            ball = Ball(Point(x,y),vx=vx,vy=vy)
            self.balls.append(ball)
            # Upgrade the game grid
            self.game_grid[y, x] = ball.number

    def get_agent_state(self) -> np.ndarray:
        self.update_game_grid()
        agent_state = self.agent_grid
        # TODO Make ONE-HOT encoding
        new_matrix = np.array([object_map[val] for row in agent_state for val in row]).flatten()


        x_control = 0 
        y_control = 0
        if (self.agent_location.x - self.food_location.x )> 0:
            x_control = 1
        if (self.agent_location.y - self.food_location.y )> 0:
            y_control = 1
        if (self.agent_location.x - self.food_location.x )< 0:
            x_control = -1
        if (self.agent_location.y - self.food_location.y )< 0:
            y_control = -1
        state = np.concatenate((new_matrix,np.array([x_control,y_control])))

        return state

    def update_agent_grid(self):
        """
        agent grid is a 5x5 grid centered on the agent. The agent can see everything 
        in this 5x5 grid.
        """
        # Clean old the agent_grid
        self.agent_grid = np.zeros((5, 5))
        
        
        offset_x = self.agent_location.x - 2
        offset_y = self.agent_location.y - 2

        # iterate on 5x5 grid
        for i in range(5):
            for j in range(5):
                real_x = offset_x + j
                real_y = offset_y + i


                if 0 <= real_x < self.w and 0 <= real_y < self.h:
                    self.agent_grid[i, j] = self.game_grid[real_y, real_x]
                else:
                    self.agent_grid[i, j] = 2  # 2 for wall or outside of the gird

    def update_game_grid(self):
        self.game_grid = np.zeros((self.w,self.h))
        # adding character
        self.game_grid[self.agent_location.y,self.agent_location.x] = 1

        # adding food
        self.game_grid[self.food_location.y,self.food_location.x] = -1

        # adding balls
        for ball in self.balls:
            self.game_grid[ball.location.y,ball.location.x] = ball.number
        
        self.update_agent_grid()

    def distance_to_food(self):
        distance = self.agent_location - self.food_location  # Manhattan distance 
        return distance
    
    def move(self,move:list):
        reward = rewards["nothing"]
        is_terminal = False 
        current_distance = self.distance_to_food()
        # Moving agent
        dx , dy = 0 , 0
        if move == 3: # Right
            dx = 1
        elif move == 2: # Left
            dx = -1
        elif move == 0: # Up
            dy = -1
        elif move == 1: # Down
            dy = 1
        elif move == 4: # Nothing
            dx = 0
            dy = 0
        new_x = self.agent_location.x + dx
        new_y = self.agent_location.y + dy

        if new_x < self.h and new_x >= 0:
            self.agent_location.x = new_x
        if new_y < self.w and new_y >= 0:
            self.agent_location.y = new_y

        if new_x > self.h or new_x <0:
            reward = rewards["wall"]
        if new_y > self.w or new_y <0:
            reward = rewards["wall"]
        

        # Moving Balls
        for ball in self.balls:
            new_x = ball.location.x + ball.vx
            new_y = ball.location.y + ball.vy

            # Check for collisions with horizontal walls
            if new_x >= self.w:  # Right wall
                ball.vx *= -1
                new_x = self.w - 2
            elif new_x < 0:  # Left wall
                ball.vx *= -1
                new_x = 1

            # Check for collisions with vertical walls
            if new_y >= self.h:  # Bottom wall
                ball.vy *= -1
                new_y = self.h - 2
            elif new_y < 0:  # Top wall
                ball.vy *= -1
                new_y = 1

            # Update ball location
            ball.location.x = new_x
            ball.location.y = new_y

            # Update ball number
            ball.number = ball.get_number()
        
        self.update_game_grid()
        self.update_agent_grid()

        # Check for food
        if self.agent_location == self.food_location:
            reward = rewards["eating_food"]
            self.score += 1
            is_terminal = False
            self.food_location = self.add_goal()
            new_state = self.get_agent_state()
            return reward,is_terminal,new_state

        # Check for collision
        for ball in self.balls:
            if self.agent_location == ball.location:
                reward = rewards["collision"]
                is_terminal = True
                new_state = self.get_agent_state()
                return reward,is_terminal,new_state
        
        # if nothing

        new_distance = self.distance_to_food()
        new_state = self.get_agent_state()
        
        if new_distance < current_distance: reward = rewards["closer"] 
        elif new_distance == current_distance : reward = rewards["nothing"]
        else: reward = rewards["far"]
        return reward,is_terminal,new_state

    def is_collision(self) -> bool:
        for ball in self.balls:
            if self.agent_location == ball.location:
                return True 
        return False

class DodgeGame(Env):
    def __init__(self,render_on:bool = False, show_grid:bool = True,
                 w:int=9, h:int=9,
                 ball_number :int=Number_of_balls):

        self.agent_state_dimension = 177 # 5x5 grid + 2 food
        self.actions = [0,1,2,3,4] 
        # 0 : Up 
        # 1 : Down 
        # 2 : Left 
        # 3 : Right
        # 4 : Nothing

        self.action_size = len(self.actions)
        
        self.w,self.h = w,h
        self.state = State(width=self.w,height=self.h)

        self.render_on = render_on # visualization with pygame

        self.show_grid = show_grid

        self.Number_of_balls = ball_number

        self.game_grids = []
        self.actions_taken = []

        if self.render_on:
            # init display
            pygame.init()
            self.display = pygame.display.set_mode((self.w*BLOCK_SIZE, self.h*BLOCK_SIZE))
            pygame.display.set_caption('Dodge Game')
            self.clock = pygame.time.Clock()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

        self.reset()

    def reset(self):
        """
        Reset the environment. Returns the new state
        """
        self.state = State(width=self.w,height=self.h)

        if self.render_on:
            self.draw_frame(self.state)

        self.game_grids = []
        self.actions_taken = []

        # Return the initial state of the grid
        return self.state.get_agent_state()
    
    def draw_frame(self,state:State=None):
        """ 
        Render the pygame environment
        """
        if state == None:
            state = self.state 
        self.display.fill(BLACK)
        # Draw the character
        pygame.draw.rect(self.display, GREEN, pygame.Rect(int(state.agent_location.x)*BLOCK_SIZE+BLOCK_SIZE/8, int(state.agent_location.y)*BLOCK_SIZE+BLOCK_SIZE/8, BLOCK_SIZE*0.75, BLOCK_SIZE*0.75))

        # Draw the goal
        goal_center = (int(state.food_location.x) *BLOCK_SIZE + BLOCK_SIZE//2, int(state.food_location.y) *BLOCK_SIZE + BLOCK_SIZE//2)
        pygame.draw.circle(self.display, YELLOW, goal_center, BLOCK_SIZE//4)
        
        # Draw the balls
        for ball in self.state.balls:
            ball_center = (int(ball.location.x) * BLOCK_SIZE + BLOCK_SIZE//2,int(ball.location.y) * BLOCK_SIZE + BLOCK_SIZE//2)
            pygame.draw.circle(self.display,ball.color,ball_center,BLOCK_SIZE//4)

        # Draw game_grid
        if self.show_grid:
            for x in range(0, self.w*BLOCK_SIZE, BLOCK_SIZE):
                pygame.draw.line(self.display, GRAY, (x, 0), (x, self.h*BLOCK_SIZE))
            for y in range(0, self.h*BLOCK_SIZE, BLOCK_SIZE):
                pygame.draw.line(self.display, GRAY, (0, y), (self.w*BLOCK_SIZE, y))
        pygame.display.flip()
        pygame.display.update()

    def step(self,move):
        self.game_grids.append(self.state.game_grid)
        self.actions_taken.append(move)
        reward,is_terminal,new_state = self.state.move(move) 
        if self.render_on:
            self.draw_frame()   
        return reward,is_terminal,new_state
    
    def draw_from_matrix(self, matrix: np.ndarray):
        """
        Draw the game environment based on the given matrix.
        
        Args:
            matrix (np.ndarray): 2D array representing the game state.
                                0: empty space, -1: food, 1: character, 3-10: balls.
        """
        self.display.fill(BLACK)  # Temizle

        # Matrisin boyutlarını al
        rows, cols = matrix.shape

        # Matrisin her bir hücresini kontrol et ve uygun şeyi çiz
        for y in range(rows):
            for x in range(cols):
                value = matrix[y, x]

                if value == 0:
                    # Boşluk, çizim yapma
                    continue
                elif value == -1:
                    # Yiyecek çiz
                    food_center = (x * BLOCK_SIZE + BLOCK_SIZE // 2, y * BLOCK_SIZE + BLOCK_SIZE // 2)
                    pygame.draw.circle(self.display, YELLOW, food_center, BLOCK_SIZE // 4)
                elif value == 1:
                    # Karakteri çiz
                    pygame.draw.rect(self.display, GREEN, pygame.Rect(x * BLOCK_SIZE + BLOCK_SIZE / 8, y * BLOCK_SIZE + BLOCK_SIZE / 8, BLOCK_SIZE * 0.75, BLOCK_SIZE * 0.75))
                elif value in range(3, 11):
                    # Top çiz
                    ball_color = RED
                    ball_center = (x * BLOCK_SIZE + BLOCK_SIZE // 2, y * BLOCK_SIZE + BLOCK_SIZE // 2)
                    pygame.draw.circle(self.display, ball_color, ball_center, BLOCK_SIZE // 4)

        # Grid çizim varsa
        if self.show_grid:
            for x in range(0, cols * BLOCK_SIZE, BLOCK_SIZE):
                pygame.draw.line(self.display, GRAY, (x, 0), (x, rows * BLOCK_SIZE))
            for y in range(0, rows * BLOCK_SIZE, BLOCK_SIZE):
                pygame.draw.line(self.display, GRAY, (0, y), (cols * BLOCK_SIZE, y))

        pygame.display.flip()
        pygame.display.update()

    def save_game(self,name,qvalues):
        a = {
            "Game States":self.game_grids,
            "Actions":self.actions_taken,
            "Qvalues":qvalues
        }
        aa = pd.DataFrame(a)
        aa.to_csv(f"{name}.csv")
    
# Test Environment
if __name__ == "__main__":
    game = DodgeGame(render_on=True, w=11, h=11,show_grid=False)
    is_terminal = False
    clock = pygame.time.Clock()  # Zamanlayıcıyı oluştur

    while not is_terminal:
        if game.render_on:
            # Olayları kontrol et
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            print(game.state.game_grid,end="\n"*10)
            keys = pygame.key.get_pressed()  # Basılı olan tuşları kontrol et
            move = None
            
            if keys[pygame.K_w]:  
                move = 0
            elif keys[pygame.K_a]:  
                move = 2
            elif keys[pygame.K_s]:  
                move = 1
            elif keys[pygame.K_d]:  
                move = 3
            else:
                move = 4

        reward,is_terminal,new_state = game.step(move)  # Karakteri hareket ettir
        game.draw_frame(game.state)
        clock.tick(3)  # FPS'i 10 olarak ayarla
    game.save_game("Oyun2")

            



