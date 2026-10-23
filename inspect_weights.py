import numpy as np

w_dict = np.load('export/weights_float32.npz')
for k in w_dict.files:
    print(k, w_dict[k].shape, w_dict[k].dtype)

calib = np.load('export/calibration.npz')
print("Calib:", calib['x'].shape, calib['y'].shape)

test = np.load('export/test.npz')
print("Test:", test['x'].shape, test['y'].shape)
