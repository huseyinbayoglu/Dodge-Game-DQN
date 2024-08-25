from game import State
from collections import deque
import random
import numpy as np

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



class Experience:
    __slots__ = ("state","action","reward","new_state","terminal")
    def __init__(self,
                 state:State,
                 action:int,
                 reward:float,
                 new_state:State,
                 terminal:bool) -> None:
        self.state = state
        self.action = action
        self.reward = reward
        self.new_state = new_state
        self.terminal = terminal 

    def visualize_experience(self):
        self.state.visualize()
        pass

    def __str__(self) -> str:
        text = "The Agent Grid: \n"
        s1 = self.state
        f1 = self.state[-2]  # 1 ise solumda -1 ise sağımda
        f2 = self.state[-1]  # 1 ise üstümde - ise altımda
        text += str(s1) + "\n"
        text += f"Solumda mı?: {f1}\n"
        text += f"Üstümde mi?: {f2}\n"
        s2 = self.new_state
        text += f"Seçilen hamle: {self.action} \t Alınan ödül: {self.reward}\t Terminal mi?: {self.terminal}"
        text += f"\n Yeni oyun durumu :\n\n {s2}"

        return text 
    
    def __repr__(self) -> str:
        return ("("+str(self.state) + " " + str(self.action) + " " + str(self.reward) + " " + str(self.new_state) + " " + str(self.terminal)+")")

    def as_sample(self) -> tuple:
        sample = (self.state,self.action,self.reward,self.new_state,self.terminal)
        return sample



class ReplayBuffer:
    def __init__(self,max_len) -> None:
        self.max_len = max_len
        self.memory = deque(maxlen=self.max_len)

    def append(self,experience:Experience):
        self.memory.append(experience)
    
    def get_sample_batch(self, batch_size: int = 32):
        return random.sample(self.memory, batch_size)

