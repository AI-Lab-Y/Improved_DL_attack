import numpy as np
from keras.models import load_model


# x shape: [n]
def uint_to_array(x, bit_len=16):
    y = np.zeros((len(x), bit_len), dtype=np.uint8)
    for j in range(bit_len):
        y[:, j] = (x >> (bit_len - 1 - j)) & 1
    return y


def generate_look_up_table(net_path=None, block_size=16):
    # load neural distinguisher
    nd = load_model(net_path)
    # get inference table
    x = np.array(range(2**block_size), dtype=np.uint32)
    new_x = uint_to_array(x, bit_len=block_size)
    y = nd.predict(new_x, batch_size=10**4, verbose=0)
    y = np.squeeze(y)
    return y


# saved_folder = './saved_model/student/0x0040-0x0/scratch/ND6/'
#
# net_path = './saved_model/student/0x0040-0x0/scratch/ND6/0.5124_14_11_student_6_distinguisher.h5'
# y = generate_look_up_table(net_path=net_path, block_size=16)
# np.save(saved_folder+'1_14_13_12_11_nd.npy', y)
#
# net_path = './saved_model/student/0x0040-0x0/scratch/ND6/0.5553_14_12_11_5_student_6_distinguisher.h5'
# y = generate_look_up_table(net_path=net_path, block_size=16)
# np.save(saved_folder+'2_14_12_11_5_nd.npy', y)
#
# net_path = './saved_model/student/0x0040-0x0/scratch/ND6/0.5953_13_12_11_10_student_6_distinguisher.h5'
# y = generate_look_up_table(net_path=net_path, block_size=16)
# np.save(saved_folder+'3_13_12_11_10_nd.npy', y)
#
# net_path = './saved_model/student/0x0040-0x0/scratch/ND6/0.6224_12_11_10_2_student_6_distinguisher.h5'
# y = generate_look_up_table(net_path=net_path, block_size=16)
# np.save(saved_folder+'4_12_11_10_2_nd.npy', y)
#
# net_path = './saved_model/student/0x0040-0x0/scratch/ND6/0.6239_12_9_student_6_distinguisher.h5'
# y = generate_look_up_table(net_path=net_path, block_size=16)
# np.save(saved_folder+'5_12_11_10_9_nd.npy', y)
#
# net_path = './saved_model/student/0x0040-0x0/scratch/ND6/0.6446_12_5_4_3_student_6_distinguisher.h5'
# y = generate_look_up_table(net_path=net_path, block_size=16)
# np.save(saved_folder+'6_12_5_4_3_nd.npy', y)
#
# net_path = './saved_model/student/0x0040-0x0/scratch/ND6/0.6683_5_2_student_6_distinguisher.h5'
# y = generate_look_up_table(net_path=net_path, block_size=16)
# np.save(saved_folder+'7_5_4_3_2_nd.npy', y)
#
# net_path = './saved_model/student/0x0040-0x0/scratch/ND6/0.6724_4_1_student_6_distinguisher.h5'
# y = generate_look_up_table(net_path=net_path, block_size=16)
# np.save(saved_folder+'8_4_3_2_1_nd.npy', y)
#
# net_path = './saved_model/student/0x0040-0x0/scratch/ND6/0.6794_11_4_3_2_student_6_distinguisher.h5'
# y = generate_look_up_table(net_path=net_path, block_size=16)
# np.save(saved_folder+'9_11_4_3_2_nd.npy', y)

#
# saved_folder = './saved_model/student/0x0040-0x0/scratch/small_net/v1_setting/'
# net_path = './saved_model/student/0x0040-0x0/scratch/small_net/v1_setting/0.5722_14_11_5_4_student_7_distinguisher.h5'
# y = generate_look_up_table(net_path=net_path, block_size=24)
# np.save(saved_folder+'0.5722_14_11_5_4_nd7.npy', y)

# saved_folder = './saved_model/student/0x0040-0x0/scratch/small_net/v1_setting/'
# net_path = saved_folder + '0.558_13_10_5_4_student_7_distinguisher.h5'
# y = generate_look_up_table(net_path=net_path, block_size=24)
# np.save(saved_folder+'0.558_13_10_5_4_nd7.npy', y)
#
# saved_folder = './saved_model/student/0x0040-0x0/scratch/small_net/v1_setting/'
# net_path = saved_folder + '0.5223_7_2_student_7_distinguisher.h5'
# y = generate_look_up_table(net_path=net_path, block_size=24)
# np.save(saved_folder+'0.5223_7_2_nd7.npy', y)
#
# saved_folder = './saved_model/student/0x0040-0x0/scratch/small_net/v1_setting/'
# net_path = saved_folder + '0.5397_14_9_student_7_distinguisher.h5'
# y = generate_look_up_table(net_path=net_path, block_size=24)
# np.save(saved_folder+'0.5397_14_9_nd7.npy', y)

saved_folder = './saved_model/student/0x0040-0x0/scratch/small_net/v1_setting/'
net_path = saved_folder + '0.5416_12_7_student_7_distinguisher.h5'
y = generate_look_up_table(net_path=net_path, block_size=24)
np.save(saved_folder+'0.5416_12_7_nd7.npy', y)

# saved_folder = './saved_model/student/0x0040-0x0/scratch/small_net/v1_setting/'
# net_path = saved_folder + '0.5224_10_8_4_2_student_7_distinguisher.h5'
# y = generate_look_up_table(net_path=net_path, block_size=24)
# np.save(saved_folder+'0.5224_10_8_4_2_nd7.npy', y)
#
# saved_folder = './saved_model/student/0x0040-0x0/scratch/small_net/v1_setting/'
# net_path = saved_folder + '0.5393_12_9_5_4_student_7_distinguisher.h5'
# y = generate_look_up_table(net_path=net_path, block_size=24)
# np.save(saved_folder+'0.5393_12_9_5_4_nd7.npy', y)

# saved_folder = './saved_model/student/0x0040-0x0/scratch/small_net/v1_setting/'
# net_path = saved_folder + '0.523_5_0_student_7_distinguisher.h5'
# y = generate_look_up_table(net_path=net_path, block_size=24)
# np.save(saved_folder+'0.523_5_0_nd7.npy', y)
#
# saved_folder = './saved_model/student/0x0040-0x0/scratch/small_net/v1_setting/'
# net_path = saved_folder + '0.523_5_0_student_7_distinguisher.h5'
# y = generate_look_up_table(net_path=net_path, block_size=24)
# np.save(saved_folder+'0.523_5_0_nd7.npy', y)
#
# saved_folder = './saved_model/student/0x0040-0x0/scratch/small_net/v1_setting/'
# net_path = saved_folder + '0.5466_12_9_3_2_student_7_distinguisher.h5'
# y = generate_look_up_table(net_path=net_path, block_size=24)
# np.save(saved_folder+'0.5466_12_9_3_2_nd7.npy', y)

# saved_folder = './saved_model/student/0x0040-0x0/scratch/small_net/v1_setting/'
# net_path = saved_folder + '0.5608_13_10_4_3_student_7_distinguisher.h5'
# y = generate_look_up_table(net_path=net_path, block_size=24)
# np.save(saved_folder+'0.5608_13_10_4_3_nd7.npy', y)

# saved_folder = './saved_model/student/0x0040-0x0/scratch/'
# net_path = saved_folder + '0.69359_12_10_4_2_student_6_distinguisher.h5'
# y = generate_look_up_table(net_path=net_path, block_size=24)
# np.save(saved_folder+'0.69359_12_10_4_2_nd6.npy', y)

# saved_folder = './saved_model/student/0x0040-0x0/scratch/'
# net_path = saved_folder + '0.6234_14_9_student_6_distinguisher.h5'
# y = generate_look_up_table(net_path=net_path, block_size=24)
# np.save(saved_folder+'0.6234_14_9_nd6.npy', y)
#
# saved_folder = './saved_model/student/0x0040-0x0/scratch/'
# net_path = saved_folder + '0.6883_5_0_student_6_distinguisher.h5'
# y = generate_look_up_table(net_path=net_path, block_size=24)
# np.save(saved_folder+'0.6883_5_0_nd6.npy', y)

# saved_folder = './saved_model/student/0x0040-0x0/scratch/'
# net_path = saved_folder + '0.5405_15_11_0_student_6_distinguisher.h5'
# y = generate_look_up_table(net_path=net_path, block_size=24)
# np.save(saved_folder+'0.5405_15_11_0_nd6.npy', y)