import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8,4.5))
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.tick_params(labelsize=32)
plt.xticks([0,1,2,3,4,5,6,7,8,9])

plt.bar([0,1,2,3,4,5,6,7,8,9], [0,0,1,0,0,0,0,0,0,0], color="#ff585d")

fig = plt.figure(figsize=(8,4.5))
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.tick_params(labelsize=32)
plt.xticks([0,1,2,3,4,5,6,7,8,9])

plt.bar([0,1,3,4,5,6,7,8,9], [0.01,0.05,0.25,0,0,0.01,0.1,0.01,0], color="#41b6e6")
plt.bar([2], [0.57], color="#ff585d")
        
fig = plt.figure(figsize=(8,4.5))
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.tick_params(labelsize=32)
plt.xticks([0,1,2], ("H&S", "TT", "DT"))

plt.bar([0,1,2], [1,0,0], width=0.2, color="#ff585d")
        
fig = plt.figure(figsize=(8,4.5))
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.tick_params(labelsize=32)
plt.xticks([0,1,2], ("H&S", "TT", "DT"))

plt.bar([1,2], [0.4,0.05], width=0.2, color="#41b6e6")
plt.bar([0], [0.55], width=0.2, color="#ff585d")