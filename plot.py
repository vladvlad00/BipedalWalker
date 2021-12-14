import matplotlib.pyplot as plt


def plot(input_name):
    with open(input_name, 'r') as file:
        lines = file.readlines()

        x_axis = []
        value_axis = []
        average_axis = []
        for line in lines:
            tokens = line.strip().replace('\t', ' ').split(' ')
            x_axis.append(int(tokens[1]))
            value_axis.append(float(tokens[6]))
            average_axis.append(float(tokens[-1]))

        plt.plot(x_axis, value_axis, label="Value")
        plt.xlabel('nr episode')
        plt.legend()
        plt.savefig(f'{input_name}_values.png')
        plt.close()

        plt.plot(x_axis, average_axis, label="Last avg interval")
        plt.xlabel('nr episode')
        plt.legend()
        plt.savefig(f'{input_name}_average.png')
        plt.close()


if __name__ == '__main__':
    plot('ddpg_minibatch_test_750.txt')