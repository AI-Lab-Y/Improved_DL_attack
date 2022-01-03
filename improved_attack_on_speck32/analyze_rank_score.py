import numpy as np

# file0 = './true_key_rank_score/7_6_0_key_rank_score_res.npy'
# file1 = './true_key_rank_score/7_6_1_key_rank_score_res.npy'

file0 = './0_key_rank_score_res.npy'
file1 = './1_key_rank_score_res.npy'

res = np.load(file1)

print(np.min(res[:, 0]), ' ', np.min(res[:, 1]))
print(np.percentile(res[:, 0], 2), ' ', np.percentile(res[:, 1], 2))
print(np.percentile(res[:, 0], 10), ' ', np.percentile(res[:, 1], 10))
print(np.percentile(res[:, 0], 25), ' ', np.percentile(res[:, 1], 25))
print(np.percentile(res[:, 0], 98), ' ', np.percentile(res[:, 1], 98))

a, b = np.percentile(res[:, 0], 10), np.percentile(res[:, 1], 10)
acc = 0
for i in range(len(res)):
    # if res[i][0] > a and res[i][1] > b:
    if res[i][0] > 10:
        acc = acc + 1
print('acc is ', acc / len(res))


res = np.load(file0)

print(np.max(res[:, 0]), ' ', np.max(res[:, 1]))
print(np.percentile(res[:, 0], 2), ' ', np.percentile(res[:, 1], 2))
print(np.percentile(res[:, 0], 10), ' ', np.percentile(res[:, 1], 10))
print(np.percentile(res[:, 0], 25), ' ', np.percentile(res[:, 1], 25))
print(np.percentile(res[:, 0], 99), ' ', np.percentile(res[:, 1], 98))

c, d = np.percentile(res[:, 0], 98), np.percentile(res[:, 1], 98)
acc = 0
for i in range(len(res)):
    # if res[i][0] > a and res[i][1] > b:
    if res[i][0] > 10:
        acc = acc + 1
print('acc is ', acc / len(res))