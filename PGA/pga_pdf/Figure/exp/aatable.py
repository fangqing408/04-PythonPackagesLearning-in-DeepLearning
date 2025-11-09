import pandas as pd

folder = "acc/"
file_name = ["ROCKET.csv", "ITime.csv", "LSTime.csv", "KDTime.csv", "KDCT.csv", "KDCR.csv"]

results = {}
for i in range(len(file_name)):
    file_name[i] = file_name[i].strip(".csv")
    results[file_name[i]] = pd.read_csv(folder + file_name[i]+".csv")
    
dataset_name = results[file_name[0]].loc[:, "dataset"].tolist()
    
f = open("table.txt", "w")

total_mean = [0]*6
total_std = [0]*6
for i in range(len(dataset_name)):
    s = dataset_name[i]
    
    mean = []
    std = []
    for _, r in results.items():
        mean.append(str(round(r.loc[i]["acc_mean"], 3)))
        std.append(str(round(r.loc[i]["acc_std"], 3)))
    
    max_mean = max(mean)
    min_std = min(std)
        
    for i in range(len(mean)):
        s += " & $"
        
        if mean[i] == max_mean:
            s += "\mathbf{" + mean[i] + "}"
            total_mean[i] += 1
        else:
            s += mean[i]
        
        s+= "\pm"
        
        if std[i] == min_std:
            s += "\mathbf{" + std[i] + "}"
            total_std[i] += 1
        else:
            s += std[i]
            
        s += "$"
    s += " \\\\"
    
    f.write(s + "\n")

s = "\hline\n" + "Total accuracy wins"
for i in range(len(total_mean)):
    s += " & $" + str(total_mean[i]) + "$"
s += " \\\\"

s += "\n" + "Total std wins"
for i in range(len(total_std)):
    s += " & $" + str(total_std[i]) + "$"
s += " \\\\"

f.write(s + "\n")

f.close()
