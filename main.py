import gym
import numpy as np
import utils
import Actor_Critic

if __name__=='__main__':
    env = gym.make('CartPole-v0')
    agent = Actor_Critic.Agent(alpha=1e-5, n_actions=env.action_space.n)
    n_games = 1000

    filename = 'cartpole.png'
    figure_file = 'plots/'+filename

    best_score = env.reward_range[0]
    score_his = []
    load_check = False

    if load_check:
        agent.load_model()
    
    for i in range(n_games):
        obser = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.get_action(obser)
            obser_, reward, done, info = env.step(action)
            score += reward

            if not load_check:
                agent.learn(obser, reward, obser_, done)

            obser = obser_
        score_his.append(score)
        avg_score = np.mean(score_his[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_check:
                agent.save_model()
        
        print(f"Episode {i} score: {score} avg score: {avg_score}")

    
    x = [i+1 for i in range(n_games)]
    utils.plot_learning_curve(x, score_his, figure_file)  


            