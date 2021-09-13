# Import
from simple_fc_model import *

# Linear Dataset ###########################################################

# linear_train_x, linear_train_y = generate_linear(n=100)
# linear_test_x, linear_test_y = generate_linear(n=100)
#
# linear_model = Model(fc1feat=3, fc2feat=3)
#
# # Model.plot_model(linear_model)
#
# linear_model.train(x=linear_train_x, y=linear_train_y, batch_size_ratio=0.1, epochs=300, lr=1, plot=True)
# linear_preds = linear_model.test(test_x=linear_test_x, test_y=linear_test_y, print_pred_values=True,
#                                  return_pred_classes=True)
#
# show_result(x=linear_test_x, y=linear_test_y, pred_y=linear_preds)
# Model.plot_model(linear_model)

# XOR Dataset #################################################################
#
# xor_train_x, xor_train_y = generate_xor_easy()
# xor_test_x, xor_test_y = generate_xor_easy()
#
# xor_model = Model(fc1feat=3, fc2feat=2)
# Model.plot_model(xor_model)
#
# xor_model.train(x=xor_train_x, y=xor_train_y, batch_size_ratio=1, epochs=10000, lr=10, display_freq=100, plot=True)
# xor_preds = xor_model.test(test_x=xor_test_x, test_y=xor_test_y, print_pred_values=True, return_pred_classes=True)
#
# show_result(x=xor_test_x, y=xor_test_y, pred_y=xor_preds)
# Model.plot_model(xor_model)

# Discussion Using Linear Dataset ###########################################################

# === LR ===
# linear_train_x, linear_train_y = generate_linear(n=100)
# linear_model = Model(fc1feat=3, fc2feat=3)
# linear_model.train_diff_lr(x=linear_train_x, y=linear_train_y, epochs=500, plot=True, batch_size_ratio=1,
#                            lr_list=[16, 8, 4, 2], display_freq=10)

# === Batch size ratio ===
# linear_train_x, linear_train_y = generate_linear(n=100)
# linear_model = Model(fc1feat=3, fc2feat=3)
# linear_model.train_diff_batch_size(x=linear_train_x, y=linear_train_y, epochs=500, lr=1, plot=True,
#                                    batch_size_ratio_list=[
#                                        1,
#                                        0.5,
#                                        0.1,
#                                        0.01
#                                    ])

# === Neurons ===
# linear_train_x, linear_train_y = generate_linear(n=100)
# linear_test_x, linear_test_y = generate_linear(n=100)
# n_hidden_layers_neurons = [
#     (2, 2),
#     (20, 2),
#     (2, 20),
# ]
# # n_hidden_layers_neurons = [
# #     (2, 3),
# #     (3, 3),
# #     (4, 3),
# #     (5, 3)
# # ]
#
# for pair in n_hidden_layers_neurons:
#     linear_model = Model(fc1feat=pair[0], fc2feat=pair[1])
#     linear_model.train(x=linear_train_x, y=linear_train_y, batch_size_ratio=0.2, epochs=200, lr=1, plot=True,
#                        plot_but_not_show=True, display_freq=50)
#     linear_model.test(test_x=linear_test_x, test_y=linear_test_y, print_pred_values=False)
# plt.show()

# Without activation function ###########################################################
# P.S. Turn sigmoid into f(x) = x
# linear_train_x, linear_train_y = generate_linear(n=100)
# linear_test_x, linear_test_y = generate_linear(n=100)
#
# linear_model = Model(fc1feat=3, fc2feat=3)
#
# linear_model.no_act_train(x=linear_train_x, y=linear_train_y, batch_size_ratio=0.1, epochs=300, lr=1, plot=True)
# linear_preds = linear_model.test(test_x=linear_test_x, test_y=linear_test_y, print_pred_values=True,
#                                  return_pred_classes=True)
#
# show_result(x=linear_test_x, y=linear_test_y, pred_y=linear_preds)
