from game import * 
from replay_buffer import ReplayBuffer,Experience
from agent import Agent
from helper import plot 

class Testing:
    def __init__(self,agent:Agent,env:Env) -> None:
        self.env = env 
        self.agent = agent 
        self.agent.min_epsilon=0
        self.agent.epsilon = 0
        self.sumRewardsEpisode=[]


    def testDQNagent(self,number_episodes:int=150,visualize:bool=False):
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

                currentState=nextState
                if step_episode >= (self.env.state.score + 1)*60:
                    print("Too long episode")
                    terminalState = True
                
            print("episode : {}, Sum of rewards {}, score {}, step {}".format(index_episode,
                                                                     np.sum(rewardsEpisode),self.env.state.score,
                                                                     step_episode))        
            self.sumRewardsEpisode.append(np.sum(rewardsEpisode)) 
            scores.append(self.env.state.score)
            mean_scores.append(np.mean(np.array(scores)))
            step_episodes.append(step_episode)
            plot(scores,mean_scores,.1,title="Testing")  
            env.save_game(name=f"Games/Test/testgame{index_episode}",qvalues=q_values_action)    

        print("Bitti")



# Test Environment
if __name__ == "__main__":
    env = DodgeGame(w=19,h=19)
    agent = Agent(env=env,model_path="Model_save_model.keras")
    DQL = Testing(agent=agent,env=env)
    DQL.testDQNagent(50)