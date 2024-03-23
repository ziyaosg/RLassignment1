import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

X = []
Y = []
X_test = []
Y_test = []


env1_goal = np.array([[-0.035, 0.142, 1.245], [-0.15, 0.142, 1.245], [-0.266, 0.142, 1.245], [-0.383, 0.142, 1.245]])
env2_goal = np.array([[-0.035, 0.042, 1.345], [-0.15, 0.042, 1.245], [-0.266, 0.142, 1.345], [-0.383, 0.042, 1.245]])
env3_goal = np.array([[-0.035, 0.042, 1.145], [-0.15, 0.142, 1.245], [-0.266, 0.142, 1.245], [-0.383, 0.042, 1.145]])
goals = np.array([env1_goal, env2_goal, env3_goal])
env1_decoy = np.array([[-0.383379, 0.165331, 1.14576], [-0.183635, 0.239076, 1.31077], [-0.153111, 0.110957, 1.29395], [0.022497, 0.180345, 1.1708]])
env2_decoy = np.array([[-0.416498, 0.078665, 1.32583], [-0.196313, 0.310306, 1.34149], [-0.133594, 0.184351, 1.34316], [-0.074792, 0.148929, 1.43017]])
env3_decoy = np.array([[-0.344683, 0.024459, 1.20518], [-0.16924, 0.226221, 1.2826], [-0.099357, 0.12819, 1.32999], [-0.024455, 0.073603, 1.2148]])
decoys = np.array([env1_decoy, env2_decoy, env3_decoy])
for i in range(2):
    for j in range(4):
        for k in range(3):
            env = i
            g = j
            traj = k + 1
            input_file_name = "../planning/joint_space_full_traj/shortest_path_env_" + str(env) + "_goal_" + str(g) + "_traj_" + str(traj) + ".txt"

            with open(input_file_name, 'r') as file:
                data = file.read().splitlines()
            file.close()

            ee_traj = []
            for lines in data:
                ee_traj.append(lines[1: -1])

            for n in range(len(ee_traj)):
                ee_pose = ee_traj[n].split(',')
                ee_pose = np.array([float(m) for m in ee_pose])
                ee_pose = np.append(ee_pose, goals[env][g])
                ee_pose = np.append(ee_pose, decoys[env])
                X.append(ee_pose)
X = np.array(X)

for i in range(2):
    for j in range(4):
        for k in range(3):
            env = i
            g = j
            traj = k + 1
            input_file_name = "../reward/path_reward_env_" + str(env) + "_goal_" + str(g) + "_traj_" + str(traj) + ".txt"

            with open(input_file_name, 'r') as file:
                data = file.read().splitlines()
            file.close()

            rewards = []
            for lines in data:
                rewards.append(lines[1: -1])

            for n in range(len(rewards)):
                reward = rewards[n].split()
                reward = np.array([float(m) for m in reward])
                Y.append(reward)
Y = np.array(Y)

for j in range(4):
    for k in range(3):
        env = 2
        g = j
        traj = k + 1
        input_file_name = "../planning/joint_space_full_traj/shortest_path_env_" + str(env) + "_goal_" + str(g) + "_traj_" + str(traj) + ".txt"

        with open(input_file_name, 'r') as file:
            data = file.read().splitlines()
        file.close()

        ee_traj = []
        for lines in data:
            ee_traj.append(lines[1: -1])

        for n in range(len(ee_traj)):
            ee_pose = ee_traj[n].split(',')
            ee_pose = np.array([float(m) for m in ee_pose])
            ee_pose = np.append(ee_pose, goals[env][g])
            ee_pose = np.append(ee_pose, decoys[env])
            X_test.append(ee_pose)

X_test = np.array(X_test)

for j in range(4):
    for k in range(3):
        env = 2
        g = j
        traj = k + 1
        input_file_name = "../reward/path_reward_env_" + str(env) + "_goal_" + str(g) + "_traj_" + str(traj) + ".txt"

        with open(input_file_name, 'r') as file:
            data = file.read().splitlines()
        file.close()

        rewards = []
        for lines in data:
            rewards.append(lines[1: -1])

        for n in range(len(rewards)):
            reward = rewards[n].split()
            reward = np.array([float(m) for m in reward])
            Y_test.append(reward)

Y_test = np.array(Y_test)

seeds = range(1, 5)
train_evaluations = []
test_evaluations = []
for seed in seeds:
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size = 0.2 * seed, random_state=seed)
    
    print(X_train.shape)
    print(X_val.shape)
    print(y_train.shape)
    print(y_val.shape)
    
    model = Sequential()
    
    model.add(Dense(32, input_dim = 22, activation ='relu'))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dense(3, activation = 'linear'))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs = 50, batch_size = 10)
    
    # Evaluate on Training Data
    train_predictions = model.predict(X_train)
    train_mse = mean_squared_error(y_train, train_predictions)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, train_predictions)
    train_r2 = r2_score(y_train, train_predictions)

    train_evaluation = [X_train.shape[0], train_mse, train_rmse, train_mae, train_r2]
    train_evaluations.append(train_evaluation)
    
    # Evaluate on Test Data
    test_predictions = model.predict(X_test)
    test_mse = mean_squared_error(Y_test, test_predictions)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(Y_test, test_predictions)
    test_r2 = r2_score(Y_test, test_predictions)

    test_evaluation = [X_train.shape[0], test_mse, test_rmse, test_mae, test_r2]
    test_evaluations.append(test_evaluation)
    
    name = str(X_train.shape[0]) + '_training_sample_model_joint_space.h5'
    model.save(name)
    output_string = 'complete ' + str(seed) + '\n'
    print(output_string)

file_path = "./training_evaluation_joint_space.txt"

with open(file_path, 'w') as file:
    for line in train_evaluations:
        file.write(str(line) + '\n')
file.close()

file_path = "./testing_evaluation_joint_space.txt"

with open(file_path, 'w') as file:
    for line in test_evaluations:
        file.write(str(line) + '\n')
file.close()
