from keras.models import Sequential 
from keras.layers import Dense 
from keras.optimizers.legacy import Adam
import random 
import numpy as np 
import tensorflow as tf 
from keras.models import load_model
from keras.losses import mean_squared_error 



actions = [
    0, # Up
    1, # Down
    2, # Left
    3, # Right
    4  # Nothing 
]

class Agent:
    def __init__(self,env,model_path:str=None,min_epsilon:float=.02,epsilon_decay:float=.998,
                 learning_rate:float=.001) -> None:
        self.env = env 
        self.epsilon = .98
        self.min_epsilon = min_epsilon 
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        self.state_dimension = self.env.agent_state_dimension
        self.actions = self.env.actions


        if model_path!=None:
            self.main_model = load_model(model_path)
            print("Model yüklendi")
        else:
            self.main_model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_network()

        self.actionsAppend = []

    def get_action(self,state):
        move = None
        # Epsilon greedy strategy
        if self.epsilon >= random.uniform(0,1):
            move = random.choice(self.actions)
            action_q_s = None
        else:
            self.epsilon = max(self.epsilon_decay*self.epsilon,self.min_epsilon)
            # Choose action with main network
            # state.reshape(1,self.env.agen_state.flatten().shape)
            action_q_s = self.main_model.predict(state.reshape(1,177),verbose=0)
            move = np.argmax(action_q_s) 

        return move,action_q_s
    
    
    def build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_dimension, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(5, activation="linear")) 
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mse",
                      metrics=["accuracy"])
        return model
    
    def update_target_network(self):
        self.target_model.set_weights(self.main_model.get_weights())



if __name__ == "__main__":
    pass