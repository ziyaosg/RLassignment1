# RL assignment 1

## This github repo is for CPSC 589 Spring '24 HW1

### Behaviors to learn

1. Go directly to the Goal
2. Go to the Goal but avoid the Decoys
3. Go to the Goal but is attracted to the Decoys

### Algoithm picked

#### [TAMER](https://www.researchgate.net/publication/220916820_Interactively_shaping_agents_via_human_reinforcement_the_TAMER_framework) framework

![pseudocode_RL](tamer/pseudocode_TAMER.png)

Tamer aims to predict human reward given the oneline human feedback, then using the reward to guide the robot's behavior. For this assignment, I would use MLP to replace the ReinModel provided in the pseudocode. Further, since we already have demonstration data, I will seperate the training and executing process. In other words, I will first do an offline training on human rewards for given demonstrations, then using the trained model to guide robot's behavior.

#### [Bayesian IRL](https://www.researchgate.net/publication/220815343_Bayesian_Inverse_Reinforcement_Learning) 

![pseudocode_IRL](bayesianIRL/Pseudocode_BayesianIRL.png)

Bayesian IRL aims to generate a probability distribution over the sapce of reward functions using Inverse Reinforcement Learning. In other words, this method focuses on inferring $P(reward | Demonstration)$. However, BIRL assumes that the expert has the attention to maximize the reward function of the given behavior, and since the provided demonstrations do not explicitly exhibit behaviors such as object avoiding, I will slightly modify the BIRL to make it suitable for my task. First, I assume $P(reward)$ and $P(D)$ is uniform, so that I only need to calculate $P(Demonstration | reward)$. Second, since I've already assigned human reward for all the demonstrations from previous task, I can approximate $P(Demonstration| reward)$ by using the expert reward (the ground truth $\hat{R}$) and the proposed reward ($\tilde{R}$). In my case, $P(Demonstration| reward) = e^{-(\hat{R} - \tilde{R})^2}$. Since the expert reward only covers states that is in the demonstration, I would need to train an MLP that could generate expert reward for all the states. Lastly, the robot will need to do online approximation of $\hat{R}$ using $\tilde{R}$, and use $\tilde{R}$ to guide its behavior.

### Feedback to the trajectories

First, I convert the joint-space poses to end-effector poses, and record it as [here](\eefPlanning)

Robot position with respect to the World: $[-0.2, -0.5, 1.021]$

End Effector position with respect to the world: [-0.2 - end_effector_pos.y, -0.5 + end_effector_pos.x, 1.021 + end_effector_pos.z]

|Env, Goal, Decoy| pos.x w.r.t the world | pos.y       | pos.z      |
|:--------------:|:---------------------:|:-----------:|:----------:|
|Env 1, Goal 1   |-0.035		 |0.142        |1.245	    |
|Env 1, Goal 2   |-0.15			 |0.142	       |1.245       |
|Env 1, Goal 3   |-0.266		 |0.142	       |1.245       |
|Env 1, Goal 4   |-0.383		 |0.142        |1.245	    |
|Env 1, Decoy 1  |-0.383379              |0.165331     |1.14576     |
|Env 1, Decoy 2  |-0.183635		 |0.239076     |1.31077     |
|Env 1, Decoy 3  |-0.153111              |0.110957     |1.29395     |
|Env 1, Decoy 4  |0.022497	  	 |0.180345     |1.1708      |
|Env 2, Goal 1   |-0.035		 |0.042        |1.345	    |
|Env 2, Goal 2   |-0.15			 |0.042	       |1.245       |
|Env 2, Goal 3   |-0.266		 |0.142	       |1.345       |
|Env 2, Goal 4   |-0.383		 |0.042        |1.245	    |
|Env 2, Decoy 1  |-0.416498              |0.078665     |1.32583     |
|Env 2, Decoy 2  |-0.196313		 |0.310306     |1.34149     |
|Env 2, Decoy 3  |-0.133594              |0.184351     |1.34316     |
|Env 2, Decoy 4  |-0.074792	  	 |0.148929     |1.43017     |
|Env 3, Goal 1   |-0.035		 |0.042        |1.145	    |
|Env 3, Goal 2   |-0.15			 |0.142	       |1.245       |
|Env 3, Goal 3   |-0.266		 |0.142	       |1.245       |
|Env 3, Goal 4   |-0.383		 |0.042        |1.145	    |
|Env 3, Decoy 1  |-0.344683              |0.024459     |1.20518     |
|Env 3, Decoy 2  |-0.16924		 |0.226221     |1.2826      |
|Env 3, Decoy 3  |-0.099357              |0.12819      |1.32999     |
|Env 3, Decoy 4  |-0.024455	  	 |0.073603     |1.2148      |

