import numpy as np

res = np.load('./key_recovery_record/naive_attack_res.npy')
num = len(res)
for i in range(num):
    k10, k11 = res[i][0], res[i][1]
    if k11 != '0x0':
        print(k10, k11)