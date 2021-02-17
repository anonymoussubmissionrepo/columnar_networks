import torch
from torch import nn
import matplotlib.pyplot as plt
plt.style.use('classic')
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()



models = ["LSTM", "GRU"]
bias = [True, False]
legend = []
for m in models:
    for b in bias:
        x = []
        y = {}
        if b:
            legend.append(m + " with bias")
        else:
            legend.append(m + " no bias")
        for count in range(0, 10):
            if m == "GRU":
                model = nn.GRU(50, 30, 1, bias=b)
            if m == "LSTM":
                model = nn.LSTM(50, 30, 1, bias=b)

            input = torch.zeros(1, 1, 50)
            hn = torch.zeros(1, 1, 30) + 1
            cn = torch.zeros(1, 1, 30) + 1
            for nam, a in model.named_parameters():

                if "bias" in nam:
                    a = a*0
                print(nam, a)
            # quit()

            pairs = []
            for a in range(0, 20):
                if a in y:
                    if m == "GRU":
                        y[a].append(torch.mean(torch.abs(hn)).item())
                    else:
                        y[a].append(torch.mean(torch.abs(cn)).item())
                else:
                    if m == "GRU":
                        y[a] = [torch.mean(torch.abs(hn)).item()]
                    else:
                        y[a] = [torch.mean(torch.abs(cn)).item()]

                if m == "GRU":
                    output, hn = model(input, hn)
                else:
                    output, (hn, cn) = model(input, (hn, cn))

        y_mean = []
        y_error = []
        for a in y:
            y_mean.append(np.mean(y[a]))
            y_error.append(np.std(y[a])/np.sqrt(len(y[a])))

        y_mean = np.array(y_mean)
        y_error = np.array(y_error)
        x = np.array(list(y.keys()))
        print(x.shape, y_mean.shape, y_error.shape)
        plt.fill_between(x, y_mean - y_error, y_mean + y_error, alpha=0.4)
        plt.plot(x, y_mean)
        plt.title("GRU")

plt.legend(legend)
plt.xlabel("No of steps")
plt.ylabel("Avg hidden state value")
plt.savefig("lstm_decay.pdf", format="pdf")

print(y_mean)
print(y_error)
