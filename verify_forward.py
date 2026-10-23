import numpy as np

def pad_same(x, kh, kw):
    # For odd kernel sizes (3x3): pad 1 top, 1 bottom, 1 left, 1 right
    ph1 = (kh - 1) // 2
    ph2 = kh - 1 - ph1
    pw1 = (kw - 1) // 2
    pw2 = kw - 1 - pw1
    return np.pad(x, ((0, 0), (ph1, ph2), (pw1, pw2), (0, 0)), mode='constant')

def conv2d_same(x, w, b):
    # x: (N, H, W, C_in)
    # w: (KH, KW, C_in, C_out)
    # b: (C_out,)
    N, H, W, C_in = x.shape
    KH, KW, _, C_out = w.shape
    x_pad = pad_same(x, KH, KW)
    out = np.zeros((N, H, W, C_out), dtype=np.float32)
    # Fast convolution using im2col or strided windows
    for i in range(H):
        for j in range(W):
            patch = x_pad[:, i:i+KH, j:j+KW, :] # (N, KH, KW, C_in)
            # patch flattened: (N, KH*KW*C_in)
            # w flattened: (KH*KW*C_in, C_out)
            out[:, i, j, :] = np.tensordot(patch, w, axes=([1, 2, 3], [0, 1, 2])) + b
    return out

def maxpool2d_same_2x2(x):
    # x: (N, H, W, C)
    N, H, W, C = x.shape
    pad_h = H % 2
    pad_w = W % 2
    if pad_h > 0 or pad_w > 0:
        x = np.pad(x, ((0,0), (0, pad_h), (0, pad_w), (0,0)), mode='constant')
    out_h = (H + pad_h) // 2
    out_w = (W + pad_w) // 2
    # reshape to pool
    x_reshaped = x.reshape(N, out_h, 2, out_w, 2, C)
    return x_reshaped.max(axis=(2, 4))

def maxpool2d_1x5(x):
    # pool (1, 5), stride 1
    # x: (N, 8, 10, C)
    N, H, W, C = x.shape
    out = np.zeros((N, H, W - 5 + 1, C), dtype=x.dtype)
    for j in range(W - 5 + 1):
        out[:, :, j, :] = x[:, :, j:j+5, :].max(axis=2)
    return out

def forward(x, weights):
    # Layer 1: Conv1 + ReLU + Pool1
    x = conv2d_same(x, weights['conv1_w'], weights['conv1_b'])
    x = np.maximum(0, x)
    x = maxpool2d_same_2x2(x)
    
    # Layer 2: Conv2 + ReLU + Pool2
    x = conv2d_same(x, weights['conv2_w'], weights['conv2_b'])
    x = np.maximum(0, x)
    x = maxpool2d_same_2x2(x)
    
    # Layer 3: Conv3 + ReLU + Pool3
    x = conv2d_same(x, weights['conv3_w'], weights['conv3_b'])
    x = np.maximum(0, x)
    x = maxpool2d_1x5(x)
    
    # Layer 4: Flatten + FC
    x = x.reshape(x.shape[0], -1)
    x = np.dot(x, weights['fc_w']) + weights['fc_b']
    return x

if __name__ == '__main__':
    weights = np.load('export/weights_float32.npz')
    calib = np.load('export/calibration.npz')
    x_test = calib['x'][:20]
    y_test = calib['y'][:20]
    logits = forward(x_test, weights)
    preds = logits.argmax(axis=1)
    print("Test on 20 samples:")
    print("Preds:", preds)
    print("True: ", y_test)
    print("Match:", np.mean(preds == y_test))
