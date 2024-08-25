from game import * 
from replay_buffer import ReplayBuffer,Experience
from agent import Agent
from helper import plot 
from keras.models import load_model


class DeepQLearning:
    def __init__(self,agent:Agent,env:Env,gamma:float=1.,
                 replay_buffer_size:int=10000,
                 min_replay_buffer:int=500,
                 updateTargetNetworkPeriod:int=2) -> None:
        self.env = env 
        self.agent = agent 
        self.gamma = gamma 
        self.replay_buffer_size = replay_buffer_size
        self.min_replay_buffer = min_replay_buffer
        self.updateTargetNetworkPeriod = updateTargetNetworkPeriod
        self.counterUpdateTargetNetwork = 0
        self.sumRewardsEpisode=[]


        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

    def trainDQNagent(self,number_episodes:int=150,visualize:bool=False,batch_size:int=16):
        scores = []
        mean_scores = []
        step_episodes = []
        for index_episode in range(1,number_episodes+1):
            # list that stores rewards per episode - this is necessary for keeping track of convergence 
            rewardsEpisode=[]
            step_episode = 0
            q_values_action = []

            # reset the environment at the beginning of every episode
            currentState = self.env.reset()
            terminalState = False

            while not terminalState:
                action,action_q_s = self.agent.get_action(currentState)
                # Store the q values to analysis
                q_values_action.append(action_q_s)
                # print(f"Oyun Durumu:\n{self.env.state.game_grid}\n Seçilen aksiyon:{action}\tq valueleri:{action_q_s}\n Agentin gördüğü:\n{currentState}")
                # here we step and return the state, reward, and boolean denoting if the state is a terminal state
                reward, terminalState,nextState  = self.env.step(action) 
                # print(f"Alınan ödül:{reward} Yeni oyun durumu:\n{self.env.state.game_grid}agentin gördüğü yeni durum\n{nextState}")         
                rewardsEpisode.append(reward)
                step_episode += 1

                experience1 = Experience(currentState,action,reward,nextState,terminalState)
                self.replay_buffer.append(experience1)
                # print(f"Experience eklendi. eklenen experience:\n{experience1}")
                if len(self.replay_buffer.memory) >= self.min_replay_buffer:
                    # sample a batch from the replay buffer
                    randomSampleBatch = self.replay_buffer.get_sample_batch(batch_size)

                    # here we form current state batch 
                    # and next state batch
                    # they are used as inputs for prediction
                    currentStateBatch=np.zeros(shape=(batch_size,self.env.agent_state_dimension)) 
                    nextStateBatch=np.zeros(shape=(batch_size,self.env.agent_state_dimension))  
                    

                    for index,experience in enumerate(randomSampleBatch):
                        # first entry of the tuple is the current state
                        currentStateBatch[index,:]=experience.state
                        # fourth entry of the tuple is the next state
                        nextStateBatch[index,:]=experience.new_state

                    # here, use the target network to predict Q-values 
                    QnextStateTargetNetwork=self.agent.target_model.predict(nextStateBatch,verbose=0)
                    # here, use the main network to predict Q-values 
                    QcurrentStateMainNetwork=self.agent.main_model.predict(currentStateBatch,verbose=0)
                    # now, we form batches for training
                    # input for training
                    inputNetwork=currentStateBatch
                    # output for training
                    outputNetwork=np.zeros(shape=(batch_size,self.env.action_size))


                    # this list will contain the actions that are selected from the batch 
                    # this list is used in my_loss_fn to define the loss-function
                    self.actionsAppend=[] 
                    #print(f"incelenen batch:\n{randomSampleBatch}\n q value tahminleri:\n{QcurrentStateMainNetwork}")
                    # prepare Label
                    for index,experience in enumerate(randomSampleBatch):
                        if experience.terminal:
                            y = experience.reward
                        else:
                            y = experience.reward + self.gamma * np.max(QnextStateTargetNetwork[index])
                            #print(f"index:{index},gama*target model tahminlerin en büyüğü:{self.gamma * np.max(QnextStateTargetNetwork[index])}")

                        self.actionsAppend.append(experience.action)
                        self.agent.actionsAppend.append(experience.action)

                        # this actually does not matter since we do not use all the entries in the cost function
                        outputNetwork[index]=QcurrentStateMainNetwork[index]
                        # this is what matters
                        outputNetwork[index,experience.action]=y
                    #print(f"Labelling yapıldı yeni q valueler\n{outputNetwork}")
                    # here, we train the network
                    self.agent.main_model.fit(inputNetwork,outputNetwork,batch_size = batch_size, verbose=0,epochs=100)     
                    #print(f"Bu batch eğitildi.Modelin yeni tahminleri:\n{self.agent.main_model.predict(currentStateBatch,verbose=0)}",end="\n"*4)
                    # increase the counter for training the target network
                    self.counterUpdateTargetNetwork+=1 
                    if (self.counterUpdateTargetNetwork>(self.updateTargetNetworkPeriod-1)):
                        # copy the weights to targetNetwork
                        self.agent.update_target_network()     
                        # reset the counter
                        self.counterUpdateTargetNetwork=0

                # set the current state for the next step
                currentState=nextState
                
            print("episode : {}, Sum of rewards {}, score {}, step {}, epsilon {}".format(index_episode,
                                                                     np.sum(rewardsEpisode),self.env.state.score,
                                                                     step_episode,round(self.agent.epsilon,4)))        
            self.sumRewardsEpisode.append(np.sum(rewardsEpisode)) 
            scores.append(self.env.state.score)
            mean_scores.append(np.mean(np.array(scores)))
            step_episodes.append(step_episode)
            plot(scores,mean_scores,.1)  
            env.save_game(name=f"Games/Train/gametrain{index_episode}",qvalues=q_values_action)    

        print("Bitti")
        self.agent.main_model.save("Model_save_model.keras")

    def keep_trainingDQN(self,model,number_episodes:int=150,visualize:bool=False,batch_size:int=16):
        self.agent.min_epsilon = 0.05
        self.agent.epsilon = 0.05
        self.agent.main_model = load_model(model)
        self.agent.update_target_network()

        scores = []
        mean_scores = []
        step_episodes = []
        for index_episode in range(1,number_episodes+1):
            # list that stores rewards per episode - this is necessary for keeping track of convergence 
            rewardsEpisode=[]
            step_episode = 0
            q_values_action = []

            # reset the environment at the beginning of every episode
            currentState = self.env.reset()
            terminalState = False

            while not terminalState:
                action,action_q_s = self.agent.get_action(currentState)
                # Store the q values to analysis
                q_values_action.append(action_q_s)
                # print(f"Oyun Durumu:\n{self.env.state.game_grid}\n Seçilen aksiyon:{action}\tq valueleri:{action_q_s}\n Agentin gördüğü:\n{currentState}")
                # here we step and return the state, reward, and boolean denoting if the state is a terminal state
                reward, terminalState,nextState  = self.env.step(action) 
                # print(f"Alınan ödül:{reward} Yeni oyun durumu:\n{self.env.state.game_grid}agentin gördüğü yeni durum\n{nextState}")         
                rewardsEpisode.append(reward)
                step_episode += 1

                experience1 = Experience(currentState,action,reward,nextState,terminalState)
                self.replay_buffer.append(experience1)
                # print(f"Experience eklendi. eklenen experience:\n{experience1}")
                if len(self.replay_buffer.memory) >= self.min_replay_buffer:
                    # sample a batch from the replay buffer
                    randomSampleBatch = self.replay_buffer.get_sample_batch(batch_size)

                    # here we form current state batch 
                    # and next state batch
                    # they are used as inputs for prediction
                    currentStateBatch=np.zeros(shape=(batch_size,self.env.agent_state_dimension)) 
                    nextStateBatch=np.zeros(shape=(batch_size,self.env.agent_state_dimension))  
                    

                    for index,experience in enumerate(randomSampleBatch):
                        # first entry of the tuple is the current state
                        currentStateBatch[index,:]=experience.state
                        # fourth entry of the tuple is the next state
                        nextStateBatch[index,:]=experience.new_state

                    # here, use the target network to predict Q-values 
                    QnextStateTargetNetwork=self.agent.target_model.predict(nextStateBatch,verbose=0)
                    # here, use the main network to predict Q-values 
                    QcurrentStateMainNetwork=self.agent.main_model.predict(currentStateBatch,verbose=0)
                    # now, we form batches for training
                    # input for training
                    inputNetwork=currentStateBatch
                    # output for training
                    outputNetwork=np.zeros(shape=(batch_size,self.env.action_size))


                    # this list will contain the actions that are selected from the batch 
                    # this list is used in my_loss_fn to define the loss-function
                    self.actionsAppend=[] 
                    #print(f"incelenen batch:\n{randomSampleBatch}\n q value tahminleri:\n{QcurrentStateMainNetwork}")
                    # prepare Label
                    for index,experience in enumerate(randomSampleBatch):
                        if experience.terminal:
                            y = experience.reward
                        else:
                            y = experience.reward + self.gamma * np.max(QnextStateTargetNetwork[index])
                            #print(f"index:{index},gama*target model tahminlerin en büyüğü:{self.gamma * np.max(QnextStateTargetNetwork[index])}")

                        self.actionsAppend.append(experience.action)
                        self.agent.actionsAppend.append(experience.action)

                        # this actually does not matter since we do not use all the entries in the cost function
                        outputNetwork[index]=QcurrentStateMainNetwork[index]
                        # this is what matters
                        outputNetwork[index,experience.action]=y
                    #print(f"Labelling yapıldı yeni q valueler\n{outputNetwork}")
                    # here, we train the network
                    self.agent.main_model.fit(inputNetwork,outputNetwork,batch_size = batch_size, verbose=0,epochs=100)     
                    #print(f"Bu batch eğitildi.Modelin yeni tahminleri:\n{self.agent.main_model.predict(currentStateBatch,verbose=0)}",end="\n"*4)
                    # increase the counter for training the target network
                    self.counterUpdateTargetNetwork+=1 
                    if (self.counterUpdateTargetNetwork>(self.updateTargetNetworkPeriod-1)):
                        # copy the weights to targetNetwork
                        self.agent.update_target_network()     
                        # reset the counter
                        self.counterUpdateTargetNetwork=0

                # set the current state for the next step
                currentState=nextState
                
            print("episode : {}, Sum of rewards {}, score {}, step {}, epsilon {}".format(index_episode,
                                                                     np.sum(rewardsEpisode),self.env.state.score,
                                                                     step_episode,round(self.agent.epsilon,4)))        
            self.sumRewardsEpisode.append(np.sum(rewardsEpisode)) 
            scores.append(self.env.state.score)
            mean_scores.append(np.mean(np.array(scores)))
            step_episodes.append(step_episode)
            plot(scores,mean_scores,.1)  
            env.save_game(name=f"Games/Train/gametrain{index_episode}",qvalues=q_values_action)    

        print("Bitti")
        self.agent.main_model.save("Model_save_model.keras")



# Test Environment
if __name__ == "__main__":
    env = DodgeGame(w=15,h=15)
    agent = Agent(env=env,learning_rate=1e-4,epsilon_decay=.9975,min_epsilon=.05)
    DQL = DeepQLearning(agent=agent,env=env, gamma=.4)
    DQL.trainDQNagent(500,batch_size=32)
    #DQL.keep_trainingDQN(model="deneme.keras",number_episodes=300,batch_size=16)