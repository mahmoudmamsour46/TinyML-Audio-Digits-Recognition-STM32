import numpy as np

q_data = np.load('export/quantized_model_int8.npz')
test_data = np.load('export/test.npz')
scale_in = 2.0 ** float(q_data['shift_in'])

target_idx = 1190 # Word 'five'
x_float = test_data['x'][target_idx:target_idx+1]
y_true = int(test_data['y'][target_idx])

from ptq_framework import forward_int8

# Extract weights and bit-shifts from saved parameters
q_weights = {
    'conv1_w': q_data['conv1_w'],
    'conv1_b': q_data['conv1_b'],
    'conv2_w': q_data['conv2_w'],
    'conv2_b': q_data['conv2_b'],
    'conv3_w': q_data['conv3_w'],
    'conv3_b': q_data['conv3_b'],
    'fc_w': q_data['fc_w'],
    'fc_b': q_data['fc_b'],
}
shifts = {
    'shift_in': int(q_data['shift_in']),
    'rs_conv1': int(q_data['rs_conv1']),
    'rs_conv2': int(q_data['rs_conv2']),
    'rs_conv3': int(q_data['rs_conv3']),
    'rs_fc': int(q_data['rs_fc']),
}

logits = forward_int8(x_float, q_weights, shifts)
pred = int(np.argmax(logits, axis=1)[0])

print(f"Sample 1190:")
print(f"True label : {y_true}")
print(f"Pred label : {pred}")
print(f"Logits     : {logits[0]}")
assert pred == y_true, "Should predict 5"
print("SUCCESS: Python INT8 predicts 5 accurately!")
