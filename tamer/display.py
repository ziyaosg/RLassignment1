import matplotlib.pyplot as plt
import numpy as np

def main(args=None):
    evaluation_metric = ['mse', 'rmse', 'mae', 'r2']

    file_path = './training_evaluation.txt'
    with open(file_path, 'r') as file:
        data = file.read().splitlines()
    file.close()
    train_eval = []
    for lines in data:
        number = [float(m) for m in lines[1:-1].split(',')]
        train_eval.append(number)
    train_eval = np.array(train_eval)

    file_path = './testing_evaluation.txt'
    with open(file_path, 'r') as file:
        data = file.read().splitlines()
    file.close()
    test_eval = []
    for lines in data:
        number = [float(m) for m in lines[1:-1].split(',')]
        test_eval.append(number)
    test_eval = np.array(test_eval)

    fig, axs = plt.subplots(4, 1, figsize=(10, 20))

    for i in range(4):
        axs[i].plot(train_eval[:, 0], train_eval[:, i+1], label="training")
        axs[i].plot(test_eval[:, 0], test_eval[:, i+1], label="testing")
        axs[i].set_xlabel("training samples")
        axs[i].set_ylabel(evaluation_metric[i])
        axs[i].set_title("plot of training samples vs evaluation metric")
        axs[i].legend()

    plt.tight_layout()

    plt.savefig("comparison_plots.png")

    plt.show()



if __name__ == '__main__':
    main()
