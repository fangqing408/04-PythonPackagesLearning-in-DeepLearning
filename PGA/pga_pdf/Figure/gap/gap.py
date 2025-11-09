import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.figure(figsize=(8,4.5))
plt.tick_params(labelsize=24)
plt.xlim(xmin=0,xmax=112)
plt.ylim(ymax=0.7)
plt.xlabel("Dataset",fontsize=28)
plt.ylabel("Gap",fontsize=28)

a = pd.read_csv("gap.csv").loc[:,"acc_gap"].values

i1 = (a<0.05)
i2 = (a>=0.05)

x1 = np.array(range(len(a[i1])))
x2 = len(a[i1]) + np.array(range(len(a[i2])))

plt.plot([len(a[i1]), len(a[i1])], [0, 1], "--", linewidth=1, color="#032030")
plt.bar(x1,a[i1], width=1, color="#41b6e6")
plt.bar(x2,a[i2], width=1, color="#ff585d")

#plt.savefig("../gap.png")





plt.figure(figsize=(8,4.5))
plt.grid(linestyle=":")
plt.tick_params(labelsize=24)
plt.xlabel("Epoch",fontsize=28)
plt.ylabel("Accuracy",fontsize=28)

case = pd.read_csv("case.csv")
y_train = case.loc[:,"train"].values
y_test = case.loc[:,"test"].values

x = np.arange(512) + 1

plt.plot(x, y_train, color="#ff585d", linewidth=3)
plt.plot(x, y_test, "--", color="#41b6e6", linewidth=3)
plt.legend(["Train", "Test"], loc="lower right", fontsize=24)

# plt.savefig("../case.png")

