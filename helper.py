import matplotlib.pyplot as plt
import random, math, ast
import numpy as np 
import pandas as pd 
from game import DodgeGame,State,Point,Ball
from time import sleep

plt.ion()

def plot(scores, mean_scores,pause:float=.1,title:str="Training..."):
    plt.clf()  # Mevcut figürü temizle
    plt.title(f'{title}')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores,label="Score")
    plt.plot(mean_scores,label="Mean Score")
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.legend()
    plt.draw()  
    plt.pause(pause)  

def grid_to_state(game_grid: np.ndarray) -> State:
    w, h = game_grid.shape
    state = State(w, h)
    state.game_grid = game_grid


    y1, x1 = np.where(game_grid == 1)

    if len(x1) > 0 and len(y1) > 0:
        agent_location = Point(int(x1[0]), int(y1[0]))
        state.agent_location = agent_location
    else:
        state.agent_location = None 


    y2, x2 = np.where(game_grid == -1)

    if len(x2) > 0 and len(y2) > 0:
        food_location = Point(int(x2[0]), int(y2[0]))
        state.food_location = food_location
    else:
        state.food_location = None 


    ball_numbers = [3, 4, 5, 6, 7, 8, 9, 10]
    balls = []

    for number in ball_numbers:
        y, x = np.where(game_grid == number)
        for (ix, iy) in zip(x, y):
            location = Point(int(ix), int(iy))
            

            if number == 3:  # up and right
                vx, vy = 1, -1
            elif number == 4:  # up and left
                vx, vy = -1, -1
            elif number == 5:  # down and right
                vx, vy = 1, 1
            elif number == 6:  # down and left
                vx, vy = -1, 1
            elif number == 7:  # vertically up
                vx, vy = 0, -1
            elif number == 8:  # vertically down
                vx, vy = 0, 1
            elif number == 9:  # horizontally left
                vx, vy = -1, 0
            elif number == 10:  # horizontally right
                vx, vy = 1, 0

            ball = Ball(location, vx, vy)
            balls.append(ball)

    # Topları state'e ekle
    state.balls = balls

    return state

actions1 = {0:"UP",
           1:"DOWN",
           2:"LEFT",
           3:"RİGHT",
           4:"NOTHİNG"}

def watch_game(game:str):
    data = pd.read_csv(game) 
    statesstr = data["Game States"].tolist()
    q_values = data["Qvalues"].tolist()
    # Her stringi bir numpy array'e dönüştürmek için
    states = []
    for matrix_str in statesstr:
        # String'i temizleyip sayıları bir listeye çevirmek
        matrix_list = [float(num) for num in matrix_str.replace('[', '').replace(']', '').split()]
        # Matrisin boyutunu hesaplamak için liste uzunluğunun karekökünü alıyoruz
        size = int(math.sqrt(len(matrix_list)))
        # Listeyi numpy array'e çevirip yeniden şekillendirmek
        matrix = np.array(matrix_list).reshape(size, size)
        states.append(matrix)
    actions = data["Actions"].tolist()
    first_state = states[0]


    env = DodgeGame(render_on=True,show_grid=False,w=size,h=size)
    env.state = grid_to_state(first_state) 



    for state, action, q_value in zip(states, actions, q_values):

        if isinstance(q_value, str):

            q_value = q_value.strip('[]')  
            q_value_list = q_value.split()  


            q_value_list = [float(num) for num in q_value_list]


            q_value = np.array(q_value_list)
            #sleep(8)
        sleep(.1)
        env.draw_from_matrix(state)
        


if __name__ == "__main__": 
    for i in range(1,501):
        sleep(.4)
        print(f"{i}. game")
        watch_game(f"Games/Test/testgame{i}.csv")

"""    env = DodgeGame(render_on=True,show_grid=True,w=7,h=7)
    sta = np.array([[0,0,0,0,0,0,-1],
                   [0,0,0,9,0,0,0],
                   [0,0,0,0,0,0,5],
                   [0,0,0,0,0,0,0],
                   [0,0,1,0,0,0,0],
                   [0,0,3,0,7,0,0],
                   [0,0,0,0,0,0,0]])
    env.draw_from_matrix(sta)
    print(sta)
    sleep(3)
    env.state = grid_to_state(sta)
    env.state.update_game_grid()
    print(f"\n\n\n{env.state.agent_grid}")
    env.draw_from_matrix(env.state.agent_grid)
    sleep(8)"""



