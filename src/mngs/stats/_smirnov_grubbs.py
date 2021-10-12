import numpy as np
from scipy import stats


def smirnov_grubbs(data, alpha=0.05):
    in_data, out_data = list(data), []
    while True:
        n = len(in_data)
        t = stats.t.isf(q=(alpha / n) / 2, df=n - 2)
        tau = (n - 1) * t / np.sqrt(n * (n - 2) + n * t * t)
        i_min, i_max = np.argmin(in_data), np.argmax(in_data)
        mu, std = np.mean(in_data), np.std(in_data, ddof=1)

        i_far = (
            i_max
            if np.abs(in_data[i_max] - mu) > np.abs(in_data[i_min] - mu)
            else i_min
        )
        tau_far = np.abs((in_data[i_far] - mu) / std)
        if tau_far < tau:
            break
        out_data.append(in_data.pop(i_far))
    return (np.array(in_data), np.array(out_data))
