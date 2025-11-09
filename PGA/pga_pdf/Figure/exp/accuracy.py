import os
import pandas as pd
from CD_Diagram import draw_cd_diagram
from CD_Diagram import cd_diagram

def selective_cd_diagram(csv_dir, fig_dir, col):
    csv_list = ["KDTime.csv", "KDCT.csv", "KDCR.csv"]
    results = pd.DataFrame(columns = ["classifier_name", "dataset_name", "accuracy"])
    
    result_KD = pd.read_csv(os.path.join(csv_dir, csv_list[0]))
    val_acc_KD = result_KD[col].values
    acc_idx = val_acc_KD < 0.80
    
    for csv in csv_list:
        result = pd.read_csv(os.path.join(csv_dir, csv))
        
        dataset_val_acc = result[["dataset", col]][acc_idx]
        classifier_name = pd.DataFrame(columns = ["classifier_name"], data = [csv[:-4]]*dataset_val_acc.shape[0])
        dataset_val_acc.reset_index(drop=True, inplace=True)
        
        perf = pd.concat([classifier_name, dataset_val_acc], axis=1)
        perf.rename(columns={'dataset': 'dataset_name', col: 'accuracy'}, inplace=True)
        
        results = results.append(perf)
        
    draw_cd_diagram(df_perf=results, labels=True, save_dir=fig_dir)
    
cd_diagram("acc", "../acc.png", "acc_mean")
selective_cd_diagram("acc", "../acc3.png", "acc_mean")