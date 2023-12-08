import matplotlib.pyplot as plt
import sys
import numpy as np

def plot_scores(high_scores):
    plt.figure(figsize=(12, 6))
    plt.plot(high_scores)
    plt.xlabel('Game #')
    plt.ylabel('Score')
    plt.savefig('out/scores.png')
    plt.close()

def plot(file):
    with open(file) as f:
        scores = []
        for line in f:
            if line.startswith("final score:"):
                score = int(line.split(":")[1].strip())
                scores.append(score)

        window_size = 10
        moving_averages = np.convolve(scores, np.ones(window_size) / window_size, mode='valid')

        plt.figure(figsize=(12, 6))
        plt.plot(scores, label='Scores')
        plt.plot(moving_averages, label=f'Moving Averages (window={window_size})')
        plt.xlabel('Game #')
        plt.ylabel('Score')
        plt.legend()
        plt.savefig('out/scores.png')
        plt.close()

if __name__ == '__main__':
    file = sys.argv[1]
    plot(file)
