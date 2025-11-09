import pandas as pd
import matplotlib.pyplot as plt

def pairwise_diagram(directory, save_dir, csv1, csv2, col):
    result1 = pd.read_csv("{}/{}".format(directory, csv1))
    result2 = pd.read_csv("{}/{}".format(directory, csv2))
    csv1 = csv1.strip(".csv")
    csv2 = csv2.strip(".csv")
    
    col1 = result1[col]
    col2 = result2[col]
    s = min(col1.min(), col2.min())
    e = max(col1.max(), col2.max())
    gap = e - s
    
    # Plot the diagram
    plt.figure(figsize=(8,4.5))
    plt.tick_params(labelsize=24)
    plt.xlabel(csv1, fontsize=24)
    plt.ylabel(csv2, fontsize=24)
    
    # texts in the diagram
    upper_left = csv2 if col=="acc_mean" or col =="acc_best" else csv1
    lower_right = csv1 if col=="acc_mean" or col =="acc_best" else csv2
    plt.text(s, e-0.1*gap, "{} better here".format(upper_left), ha="left", fontsize=24)
    plt.text(e, s+0.05*gap, "{} better here".format(lower_right), ha="right", fontsize=24)
    
    # diagonal line
    plt.plot([s, e], [s, e], linewidth = 4, color="#41b6e6")
    
    # points
    plt.scatter(col1, col2, color="#ff585d", marker=".", s = 500)
    # plt.show()
    
    plt.savefig(save_dir, bbox_inches='tight')

if __name__ == "__main__":
    pairwise_diagram("./pair_comparison", "../testt_kdcr_rocket.png", "KDCTime.csv", "ROCKET.csv", "time_val")
