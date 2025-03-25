import numpy as np
import matplotlib.pyplot as plt




def plot_isotherms(true, pred, p):

    n = len(true)

    # create subplots with 2 rows and n/2 columns
    # each plot should be square
    fig, axs = plt.subplots(n//2, 2, figsize=(10, 5*n//2))

    for i in range(n):
        ax = axs[i//2, i%2]
        ax.plot(p, true[i], label='True')
        ax.plot(p, pred[i], label='Predicted')
        ax.set_title(f'Isotherm {i+1}')
        ax.set_xlabel('Log Pressure (bar)')
        ax.set_ylabel('Uptake (mol/kg)')
        ax.set_ylim(0, 3.5)
        ax.legend()

    plt.tight_layout()

    return fig