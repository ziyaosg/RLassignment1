import numpy as np

# Function to calculate the mean square root difference
def square_root_difference(values, reference):
    return np.sqrt(np.sum(np.square(np.array(values) - np.array(reference))))

def calcualte_reward(pose, goal, decoy):
    weight1 = np.array([1, 0, 0, 0, 0])
    weight2 = np.array([1, -0.2, -0.2, -0.2, -0.2])
    weight3 = np.array([1, 0.2, 0.2, 0.2, 0.2])
    distance_goal = square_root_difference(pose, goal)
    distance_decoy1 = square_root_difference(pose, decoy[0])
    distance_decoy2 = square_root_difference(pose, decoy[1])
    distance_decoy3 = square_root_difference(pose, decoy[2])
    distance_decoy4 = square_root_difference(pose, decoy[3])
    raw_score = np.array([distance_goal, distance_decoy1, distance_decoy2, distance_decoy3, distance_decoy4])
    return np.array([np.exp(- np.dot(weight1, raw_score)), np.exp(- np.dot(weight2, raw_score)), np.exp(- np.dot(weight3, raw_score))])

env1_goal = np.array([[-0.035, 0.142, 1.245], [-0.15, 0.142, 1.245], [-0.266, 0.142, 1.245], [-0.383, 0.142, 1.245]])
env2_goal = np.array([[-0.035, 0.042, 1.345], [-0.15, 0.042, 1.245], [-0.266, 0.142, 1.345], [-0.383, 0.042, 1.245]])
env3_goal = np.array([[-0.035, 0.042, 1.145], [-0.15, 0.142, 1.245], [-0.266, 0.142, 1.245], [-0.383, 0.042, 1.145]])
goals = np.array([env1_goal, env2_goal, env3_goal])
env1_decoy = np.array([[-0.383379, 0.165331, 1.14576], [-0.183635, 0.239076, 1.31077], [-0.153111, 0.110957, 1.29395], [0.022497, 0.180345, 1.1708]])
env2_decoy = np.array([[-0.416498, 0.078665, 1.32583], [-0.196313, 0.310306, 1.34149], [-0.133594, 0.184351, 1.34316], [-0.074792, 0.148929, 1.43017]])
env3_decoy = np.array([[-0.344683, 0.024459, 1.20518], [-0.16924, 0.226221, 1.2826], [-0.099357, 0.12819, 1.32999], [-0.024455, 0.073603, 1.2148]])
decoys = np.array([env1_decoy, env2_decoy, env3_decoy])

for i in range(3):
    for j in range(4):
        for k in range(3):
            env = i
            g = j
            traj = k + 1

            input_file_name = "../eefPlanning/shortest_path_env_" + str(env) + "_goal_" + str(g) + "_traj_" + str(traj) + "_eef.txt"
            output_file_name = "../reward/path_reward_env_" + str(env) + "_goal_" + str(g) + "_traj_" + str(traj) + ".txt"

            with open(input_file_name, 'r') as file:
                data = file.read().splitlines()
            file.close()

            jt_traj = []
            for lines in data:
                jt_traj.append(lines[1:-1])

            with open(output_file_name, 'w') as file:
                for n in range(len(jt_traj)):
                    ee_pose = jt_traj[n].split(',')
                    ee_pose = [float(m) for m in ee_pose]
                    ee_pose = [-0.2-ee_pose[1], -0.5+ee_pose[0], 1.021 + ee_pose[2]]
                    reward = calcualte_reward(ee_pose, goals[i][j], decoys[i])
                    file.write(str(reward) + '\n')
            message = "reward assgined to shortest_path_env_"+ str(env) + "_goal_" + str(g) + "_traj_" + str(traj) + " complete \n"
            print(message)
            file.close()
