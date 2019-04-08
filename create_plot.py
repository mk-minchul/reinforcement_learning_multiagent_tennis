from matplotlib import pyplot as plt

def plot_avg_reward(path):

    with open(path, "r") as f:
        data = f.readlines()
        epochs = []
        scores = []
        for line in data[1:]:
            epoch, score = line.strip().split(",")
            epochs.append(int(epoch))
            scores.append(float(score))

    fig, ax = plt.subplots()
    ax.plot(epochs, scores)
    ax.axhline(0.5, color="red")
    ax.set_xlabel("number of episodes")
    ax.set_ylabel("avg score of 100 episodes")

    ax.set_title("Averge Score Plot for Tennis Env")
    plt.savefig(path.replace("txt","png"))

if __name__ == '__main__':
    path = "experiments/exp1_2019-04-07_15:20:11/scores.txt"
    plot_avg_reward(path)