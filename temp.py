import torch
from utils.setup import GetCustomProteinDatasetPadded, GetCVProteins
import torch.nn as nn
import matplotlib.pyplot as plt
import utils.metrics_utils as mu
import numpy as np
from sklearn import metrics
import torch.optim as optim

encode_length = 1500

# Create the dataset cunstructor (use encode_length to set the dimension of the encoded proteins)
CustomProteinDataset = GetCustomProteinDatasetPadded(encode_length)
CVProteins = GetCVProteins()


# Model
class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.channels = 512
        self.length = 1500
        self.hidden1 = 128
        batchnorm = nn.BatchNorm1d

        activation_fn = nn.ReLU

        self.net = nn.Sequential(
            nn.Conv1d(self.channels, self.hidden1, 9, padding=4),
            batchnorm(self.hidden1),
            activation_fn(),
            nn.Conv1d(self.hidden1, self.num_classes, 7, padding=3),
        )

    def forward(self, x):
        return self.net(x)


# Accuracy
def accuracy(target, pred):
    return metrics.accuracy_score(
        target.detach().cpu().numpy(), pred.detach().cpu().numpy()
    )


print_error_type_pairs = False


# Initiate the model
n_unique_labels = 7
model = Model(n_unique_labels)

# Loss function
loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)


# Setup datasets
cv0Indices = CVProteins["cv0"]
train_dataset = CustomProteinDataset(cv0Indices[0:553])
test_dataset = CustomProteinDataset(cv0Indices[553:691])

batch_size = 32
# define a data loader to iterate the dataset
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False
)

num_epochs = 20
validation_every_steps = 50

step = 0
model.train()

train_accuracies = []
valid_accuracies = []


step_history = []
type_accuracy_history = []

tm_type_accuracy_history = []
sptm_type_accuracy_history = []
sp_type_accuracy_history = []
glob_type_accuracy_history = []
beta_type_accuracy_history = []

tm_topology_accuracy_history = []
sptm_topology_accuracy_history = []
sp_topology_accuracy_history = []
glob_topology_accuracy_history = []
beta_topology_accuracy_history = []

print("Running Training loop")

for epoch in range(num_epochs):
    train_accuracies_batches = []

    for batch in train_loader:
        inputs, targets = batch
        output = model(inputs[:, :-1, :])

        loss = loss_fn(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Increment step counter
        step += 1

        # Compute accuracy
        predictions = output.max(1)[1]

        # Calculate accuracy for each protein in batch
        # prediction_labels_list = []
        # target_labels_list = []
        for idx in range(predictions.shape[0]):
            target_len = int(torch.sum(inputs[idx, -1, :]))
            train_accuracies_batches.append(
                accuracy(targets[idx][0:target_len], predictions[idx][0:target_len])
            )

        if step % validation_every_steps == 0:
            # Append everage training accuracy to list
            train_accuracies.append(np.mean(train_accuracies_batches))

            train_accuracies_batches = []

            # Compite accuracies on validation set
            valid_accuracies_batches = []

            # For confusion matrix
            prediction_labels_list = []
            target_labels_list = []

            with torch.no_grad():
                model.eval()
                # Change this to test_loader once this exists
                for batch_test in test_loader:
                    inputs, targets = batch_test

                    # inputs, targets = inputs.to(DEVICE), targets.to(DEVICE) # Probably change this!!
                    output = model(inputs[:, :-1, :])
                    loss = loss_fn(output, targets)

                    predictions = output.max(1)[1]

                    for idx in range(predictions.shape[0]):
                        # target_len represents the length of the targets non padded length
                        target_len = int(torch.sum(inputs[idx, -1, :]))

                        valid_accuracies_batches.append(
                            accuracy(
                                targets[idx][0:target_len],
                                predictions[idx][0:target_len],
                            )
                        )

                        prediction_labels_list += [predictions[idx][0:target_len]]
                        target_labels_list += [targets[idx][0:target_len]]

                    # Multiply by len(x) because the final batch of DataLoader may be smaller (drop_last=False).
                    # valid_accuracies_batches.append(
                    #     accuracy(targets[0:500], predictions[0:500]) * len(inputs)
                    # )

                    model.train()

            # Calcucalte and prind confusion matrix

            valid_accuracies.append(
                np.sum(valid_accuracies_batches) / len(test_dataset)
            )
            print(f"Step {step:<5}")
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

            # Store values for plot
            step_history += [step]
            type_accuracy_history += [type_accuracy]

            tm_type_accuracy_history += [detailed_type_accuracy["tm"]["type"]]
            sptm_type_accuracy_history += [detailed_type_accuracy["sptm"]["type"]]
            sp_type_accuracy_history += [detailed_type_accuracy["sp"]["type"]]
            glob_type_accuracy_history += [detailed_type_accuracy["glob"]["type"]]
            beta_type_accuracy_history += [detailed_type_accuracy["beta"]["type"]]

            tm_topology_accuracy_history += [detailed_type_accuracy["tm"]["topology"]]
            sptm_topology_accuracy_history += [
                detailed_type_accuracy["sptm"]["topology"]
            ]
            sp_topology_accuracy_history += [detailed_type_accuracy["sp"]["topology"]]
            glob_topology_accuracy_history += [
                detailed_type_accuracy["glob"]["topology"]
            ]
            beta_topology_accuracy_history += [
                detailed_type_accuracy["beta"]["topology"]
            ]


print("Done training.")
