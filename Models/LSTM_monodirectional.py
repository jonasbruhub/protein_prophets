# Inspiration from
# https://wandb.ai/sauravmaheshkar/LSTM-PyTorch/reports/Using-LSTM-in-PyTorch-A-Tutorial-With-Examples--VmlldzoxMDA2NTA5?fbclid=IwAR0RX_QQCU9RvRUbL3bbgbcD_EiTCPqATgdUZ2nqNM1m-xLQZJf1ibJQiYk

import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch
import torch.nn as nn
import numpy as np

from utils.setup import GetCustomProteinDatasetPadded, GetCVProteins
import utils.metrics_utils as mu
from sklearn import metrics
import torch.optim as optim

encode_length = 1500
print_error_type_pairs = False

# Create the dataset cunstructor (use encode_length to set the dimension of the encoded proteins)
CustomProteinDataset = GetCustomProteinDatasetPadded(encode_length)
CVProteins = GetCVProteins()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("using device:", device)


# Model
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()

        self.input_d = 512
        self.output_d = 7
        self.hidden_dim = 128
        self.layer_dim = 1

        # LSTM model
        self.lstm = nn.LSTM(
            self.input_d, self.hidden_dim, self.layer_dim, batch_first=True
        )

        self.linear = nn.Linear(self.hidden_dim, self.output_d)

    def forward(self, x):
        hidden0 = (
            torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
            .requires_grad_()
            .to(device)
        )

        cell0 = (
            torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
            .requires_grad_()
            .to(device)
        )

        output, (hidden_n, cell_n) = self.lstm(x, (hidden0.detach(), cell0.detach()))

        output = self.linear(output[:, :, :])

        return output


# Accuracy
def accuracy(target, pred):
    return metrics.accuracy_score(
        target.detach().cpu().numpy(), pred.detach().cpu().numpy()
    )


# Given a CNN model, a training set and a validation set, return a trained model
def train_model(model, train_dataset, validation_dataset):
    # Loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    batch_size = 32

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    num_epochs = 100
    validation_every_steps = 50

    step = 0
    model.train()

    train_accuracies = []
    valid_accuracies = []

    for epoch in range(num_epochs):
        train_accuracies_batches = []

        for batch in train_loader:
            # Inputs are [batch_size, 512, 1500]: 1500 long (padded) proteins, encoded using esm to get 512 latent variables
            inputs_train, targets_train = batch
            inputs_train, targets_train = inputs_train.to(device), targets_train.to(
                device
            )

            model.zero_grad()
            output_train = model(inputs_train[:, :-1, :].permute(0, 2, 1)).permute(
                0, 2, 1
            )
            loss = loss_fn(output_train, targets_train)
            loss.backward()
            optimizer.step()

            # Increment step counter
            step += 1

            # Compute accuracy
            predictions_train = output_train.max(1)[1]

            # Calculate accuracy for each protein in batch
            for idx in range(predictions_train.shape[0]):
                target_len = int(torch.sum(inputs_train[idx, -1, :]))
                train_accuracies_batches.append(
                    accuracy(
                        targets_train[idx][:target_len],
                        predictions_train[idx][:target_len],
                    )
                )

            if step % validation_every_steps == 0:
                # Append average training accuracy to list
                train_accuracies.append(np.mean(train_accuracies_batches))

                train_accuracies_batches = []

                # Compute accuracies on validation set
                validation_accuracies_batches = []

                prediction_labels_list = []
                target_labels_list = []

                with torch.no_grad():
                    model.eval()

                    for batch_val in validation_loader:
                        inputs_val, targets_val = batch_val
                        inputs_val, targets_val = inputs_val.to(device), targets_val.to(
                            device
                        )
                        output_val = model(
                            inputs_val[:, :-1, :].permute(0, 2, 1)
                        ).permute(0, 2, 1)

                        predictions_val = output_val.max(1)[1]

                        for idx in range(predictions_val.shape[0]):
                            target_len = int(torch.sum(inputs_val[idx, -1, :]))

                            validation_accuracies_batches.append(
                                accuracy(
                                    targets_val[idx][0:target_len],
                                    predictions_val[idx][0:target_len],
                                )
                            )
                            prediction_labels_list += [
                                predictions_val[idx][0:target_len]
                            ]
                            target_labels_list += [targets_val[idx][0:target_len]]

                    model.train()

                valid_accuracies.append(
                    np.sum(validation_accuracies_batches) / len(validation_dataset)
                )

                print(f"  training accuracy:    {train_accuracies[-1]}")
                print(f"  test accuracy:        {valid_accuracies[-1]}")

                # Extra accuracies
                (
                    error_type_pairs,
                    confusion_matrix,
                    type_accuracy,
                    detailed_type_accuracy,
                ) = mu.confusionMatrix(prediction_labels_list, target_labels_list)

                # type accuracy is average of per type, topology accuracy
                print(f"  type accuracy (test): {type_accuracy}")

                # detailed type accuracies
                for key in detailed_type_accuracy.keys():
                    print(f"  {key}")
                    for field in detailed_type_accuracy[key].keys():
                        print(f"    {field:<9}: {detailed_type_accuracy[key][field]}")

                print(confusion_matrix)

                if print_error_type_pairs:
                    for error_pair in error_type_pairs:
                        print("  Predicted topology:", error_pair["predicted topology"])
                        print("  Target topology:   ", error_pair["target topology"])

                print("")

    print("Done training a model.")

    return model


def test_model(model, test_dataset):
    # Loss function

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_dataset.__len__(),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    # Compute accuracies on validation set
    test_accuracies = []
    prediction_labels_list = []
    target_labels_list = []

    with torch.no_grad():
        model.eval()
        for batch in test_loader:
            inputs, targets = batch
            inputs, tragets = inputs.to(device), targets.to(device)
            output = model(inputs[:, :-1, :].permute(0, 2, 1)).permute(0, 2, 1)

            predictions = output.max(1)[1]

            for idx in range(predictions.shape[0]):
                target_len = int(torch.sum(inputs[idx, -1, :]))
                test_accuracies.append(
                    accuracy(
                        targets[idx][0:target_len],
                        predictions[idx][0:target_len],
                    )
                )
                prediction_labels_list += [predictions[idx][0:target_len]]
                target_labels_list += [targets[idx][0:target_len]]

    print("  Test accuracy: " + str(np.sum(test_accuracies) / len(test_dataset)))

    # Extra accuracies
    (
        error_type_pairs,
        confusion_matrix,
        type_accuracy,
        detailed_type_accuracy,
    ) = mu.confusionMatrix(prediction_labels_list, target_labels_list)

    # type accuracy is average of per type, topology accuracy
    print(f"  type accuracy (test): {type_accuracy}")

    # detailed type accuracies
    for key in detailed_type_accuracy.keys():
        print(f"  {key}")
        for field in detailed_type_accuracy[key].keys():
            print(f"    {field:<9}: {detailed_type_accuracy[key][field]}")

    print(confusion_matrix)

    if print_error_type_pairs:
        for error_pair in error_type_pairs:
            print("  Predicted topology:", error_pair["predicted topology"])
            print("  Target topology:   ", error_pair["target topology"])


n_cv = CVProteins.keys().__len__()
n_unique_labels = 7


for loop in range(1):
    print("---------------------------------------------------------------------------")
    print(
        f"-                           Loop {loop:<8}                                 -"
    )
    print(
        f"-  train sets: cv{str((loop + 0) % 5):<1}, cv{str((loop + 1) % 5):<1}, cv{str((loop + 2) % 5):<1}"
    )
    print(f"-  validation sets: cv{str((loop + 3) % 5):<1}")
    print(f"-  test sets: cv{str((loop + 4) % 5):<1}")
    print("---------------------------------------------------------------------------")

    train_dataset = CustomProteinDataset(
        CVProteins["cv" + str((loop + 0) % 5)]
        + CVProteins["cv" + str((loop + 1) % 5)]
        + CVProteins["cv" + str((loop + 2) % 5)]
    )
    validation_dataset = CustomProteinDataset(CVProteins["cv" + str((loop + 3) % 5)])
    test_dataset = CustomProteinDataset(CVProteins["cv" + str((loop + 4) % 5)])

    model = LSTMModel().to(device)
    trained_model = train_model(model, train_dataset, validation_dataset)
    test_model(trained_model, test_dataset)
