from CD_Diagram import cd_diagram

cd_diagram("epoch", "../epoch.png", "acc_mean")
cd_diagram("lr", "../lr.png", "acc_mean")
cd_diagram("LS_e", "../LS_e.png", "acc_mean")
cd_diagram("KD/e", "../KD_e.png", "acc_mean")
cd_diagram("KD/T", "../KD_t.png", "acc_mean")