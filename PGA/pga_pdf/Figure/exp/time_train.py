from Pair_Diagram import pairwise_diagram

res_dir = "./pair_comparison"

pairwise_diagram(res_dir, "../traint_kdcr_rocket.png", "KDCTime.csv", "ROCKET.csv", "time_train")
pairwise_diagram(res_dir, "../traint_kdcr_itime.png", "KDCTime.csv", "ITime.csv", "time_train")
pairwise_diagram(res_dir, "../traint_kdcr_lstime.png", "KDCTime.csv", "LSTime6.csv", "time_train")
pairwise_diagram(res_dir, "../traint_kdcr_kdtime.png", "KDCTime.csv", "KDTime.csv", "time_train")
