import student_net as tn
import speck as sp


sp.check_testvector()
# selected_bits = [14, 13, 12, 11, 10, 9, 5, 4, 3, 2, 1, 0]   # acc = 0.7796
# selected_bits = [14, 13, 12, 11, 10, 4, 3, 2, 1, 0]   # acc = 0.73468
# selected_bits = [14, 13, 12, 11, 10, 5, 4, 3, 2, 1, 0]   # acc = 0.7671
# selected_bits = [14, 13, 12, 11, 10, 5, 4, 3, 2, 1]     # acc = 0.7604
# selected_bits = [15, 14, 13, 12, 11, 0]     # acc =
# nr = 6
# model_folder = './saved_model/student/0x0040-0x0/'
# tn.train_speck_distinguisher(10, num_rounds=nr, depth=1, diff=(0x0040, 0), bits=selected_bits, folder=model_folder)

# selected_bits = [14, 13, 12, 11, 10, 9, 5, 4, 3, 2]
# nr = 7
# model_folder = './saved_model/student/0x0040-0x0/'
# tn.train_speck_distinguisher(20, num_rounds=nr, depth=1, diff=(0x0040, 0), bits=selected_bits, folder=model_folder)

selected_bits = [15, 14, 13, 2, 1, 0]   # acc =
nr = 7
model_folder = './saved_model/student/0x0040-0x0/'
tn.train_speck_distinguisher(10, num_rounds=nr, depth=1, diff=(0x0040, 0), bits=selected_bits, folder=model_folder)

