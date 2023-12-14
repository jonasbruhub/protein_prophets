"""
metric_utils.py
===============

Code provided by Jeppe Hallgren.

All functions were written in pytorch 1.5 - best if you check whether 
there are any changes/warnings that you should consider for pytorch 2.0+.
"""
import torch

# import json
from typing import List, Union, Dict

# The following are the label mapping is used in the metrics.
LABELS: Dict[str, int] = {"I": 0, "O": 1, "P": 2, "S": 3, "M": 4, "B": 5, "X": -1}
LABELS_INV = {0: "I", 1: "O", 2: "P", 3: "S", 4: "M", 5: "B", -1: "X"}


def EncodeLabel(label):
    # given label string of IOPSMBX, convert/encode to integers array
    return [LABELS[lab] for lab in label]


# Helper functions
def label_list_to_topology(
    labels: Union[List[int], torch.Tensor]
) -> List[torch.Tensor]:
    """
    Converts a list of per-position labels to a topology representation.
    This maps every sequence to list of where each new symbol start (the topology), e.g. AAABBBBCCC -> [(0,A),(3, B)(7,C)]

    Parameters
    ----------
    labels : list or torch.Tensor of ints
        List of labels.

    Returns
    -------
    list of torch.Tensor
        List of tensors that represents the topology.
    """

    if isinstance(labels, list):
        labels = torch.LongTensor(labels)

    if isinstance(labels, torch.Tensor):
        zero_tensor = torch.LongTensor([0])
        if labels.is_cuda:
            zero_tensor = zero_tensor.cuda()

        unique, count = torch.unique_consecutive(labels, return_counts=True)
        top_list = [torch.cat((zero_tensor, labels[0:1]))]
        prev_count = 0
        i = 0
        for _ in unique.split(1):
            if i == 0:
                i += 1
                continue
            prev_count += count[i - 1]
            top_list.append(torch.cat((prev_count.view(1), unique[i].view(1))))
            i += 1
        return top_list


def is_topologies_equal(topology_a, topology_b, minimum_seqment_overlap=5):
    """
    Checks whether two topologies are equal.
    E.g. [(0,A),(3, B)(7,C)]  is the same as [(0,A),(4, B)(7,C)]
    But not the same as [(0,A),(3, C)(7,B)]

    Parameters
    ----------
    topology_a : list of torch.Tensor
        First topology. See label_list_to_topology.
    topology_b : list of torch.Tensor
        Second topology. See label_list_to_topology.
    minimum_seqment_overlap : int
        Minimum overlap between two segments to be considered equal.

    Returns
    -------
    bool
        True if topologies are equal, False otherwise.
    """

    if isinstance(topology_a[0], torch.Tensor):
        topology_a = list([a.cpu().numpy() for a in topology_a])
    if isinstance(topology_b[0], torch.Tensor):
        topology_b = list([b.cpu().numpy() for b in topology_b])
    if len(topology_a) != len(topology_b):
        return False

    # Fejl i denne, rettet til enumerate topology_a[:-1], så ikke tager sidste med
    for idx, (_position_a, label_a) in enumerate(topology_a[:-1]):
        if label_a != topology_b[idx][1]:
            if label_a in (1, 2) and topology_b[idx][1] in (1, 2):  # assume O == P
                continue
            else:
                return False
        if label_a in (3, 4, 5):
            overlap_segment_start = max(topology_a[idx][0], topology_b[idx][0])
            overlap_segment_end = min(topology_a[idx + 1][0], topology_b[idx + 1][0])
            if label_a == 5:
                # Set minimum segment overlap to 3 for Beta regions
                minimum_seqment_overlap = 3
            if overlap_segment_end - overlap_segment_start < minimum_seqment_overlap:
                return False
    return True


def calculate_acc(correct, total):
    total = total.float()
    correct = correct.float()
    if correct == 0.0:
        return 0
    if total == 0.0:
        return 1
    return correct / total


def type_from_labels(label):
    """
    Function that determines the protein type from labels

    Dimension of each label:
    (len_of_longenst_protein_in_batch)

    # Residue class
    0 (I) = inside cell/cytosol
    1 (O) = Outside cell/lumen of ER/Golgi/lysosomes
    2 (B) = beta membrane
    3 (S) = signal peptide
    4 (M) = alpha membrane
    5 (P) = periplasm

    B in the label sequence -> beta
    I only -> globular
    Both S and M -> SP + alpha(TM)
    M -> alpha(TM)
    S -> signal peptide

    # Protein type class
    0 = TM
    1 = SP + TM
    2 = SP
    3 = GLOBULAR
    4 = BETA
    """

    if 2 in label:
        return 4

    if all(e == 0 for e in label):
        return 3

    if (3 in label) and (4 in label):
        return 1

    if 3 in label:
        return 2

    if 4 in label:
        return 0

    if all(e == 0 or e == -1 for e in label):
        return 3

    return None


# NOTE the following was taken directly from the DeepTMHMM codebase.
# I think it is a good idea to write your own code instead.
# It is here as a reference for how the published DeepTMHMM accuracies
# were calculate using true/predicted types and true/predicted topologies.


def confusionMatrix(predicted_label_list, target_label_list):
    # predicted_label_list   : list of 1D-list of aminoacid label predictions (int)
    # target_label_list      : list of 1D-list of target aminoacid labels

    confusion_matrix = torch.zeros((5, 7), dtype=torch.int64)
    error_type_pairs = []

    for idx in range(predicted_label_list.__len__()):
        predicted_label = predicted_label_list[idx]
        target_label = target_label_list[idx]

        predicted_type = type_from_labels(predicted_label)
        predicted_topology = label_list_to_topology(predicted_label)

        target_type = type_from_labels(target_label)
        target_topology = label_list_to_topology(target_label)

        if predicted_type == None:
            # Decode label into characters
            predicted_letters = "".join(
                [LABELS_INV[int(lab)] for lab in predicted_label]
            )
            target_letters = "".join([LABELS_INV[int(lab)] for lab in target_label])
            error_type_pairs += [
                {
                    "predicted topology": predicted_letters,
                    "target topology": target_letters,
                }
            ]
            confusion_matrix[target_type][6] += 1
            continue

        if target_type == None:
            raise Exception("target_type could not be calculated")

        prediction_topology_match = is_topologies_equal(
            target_topology, predicted_topology, 5
        )

        if (target_type == predicted_type) and (
            target_type == 2 or target_type == 3 or prediction_topology_match
        ):
            # if we guessed the type right for SP+GLOB or GLOB,
            # count the topology as correct
            confusion_matrix[target_type][5] += 1
        else:
            confusion_matrix[target_type][predicted_type] += 1

    # Metrics
    type_correct_ratio = (
        calculate_acc(
            confusion_matrix[0][0] + confusion_matrix[0][5],
            confusion_matrix[0].sum(),
        )
        + calculate_acc(
            confusion_matrix[1][1] + confusion_matrix[1][5],
            confusion_matrix[1].sum(),
        )
        + calculate_acc(
            confusion_matrix[2][2] + confusion_matrix[2][5],
            confusion_matrix[2].sum(),
        )
        + calculate_acc(
            confusion_matrix[3][3] + confusion_matrix[3][5],
            confusion_matrix[3].sum(),
        )
        + calculate_acc(
            confusion_matrix[4][4] + confusion_matrix[4][5],
            confusion_matrix[4].sum(),
        )
    )
    type_accuracy = type_correct_ratio / 5

    tm_accuracy = calculate_acc(confusion_matrix[0][5], confusion_matrix[0].sum())
    sptm_accuracy = calculate_acc(confusion_matrix[1][5], confusion_matrix[1].sum())
    sp_accuracy = calculate_acc(confusion_matrix[2][5], confusion_matrix[2].sum())
    glob_accuracy = calculate_acc(confusion_matrix[3][5], confusion_matrix[3].sum())
    beta_accuracy = calculate_acc(confusion_matrix[4][5], confusion_matrix[4].sum())

    tm_type_acc = calculate_acc(
        confusion_matrix[0][0] + confusion_matrix[0][5],
        confusion_matrix[0].sum(),
    )
    tm_sp_type_acc = calculate_acc(
        confusion_matrix[1][1] + confusion_matrix[1][5],
        confusion_matrix[1].sum(),
    )
    sp_type_acc = calculate_acc(
        confusion_matrix[2][2] + confusion_matrix[2][5],
        confusion_matrix[2].sum(),
    )
    glob_type_acc = calculate_acc(
        confusion_matrix[3][3] + confusion_matrix[3][5],
        confusion_matrix[3].sum(),
    )
    beta_type_acc = calculate_acc(
        confusion_matrix[4][4] + confusion_matrix[4][5],
        confusion_matrix[4].sum(),
    )

    return (
        error_type_pairs,
        confusion_matrix,
        type_accuracy,
        {
            "tm": {"type": tm_type_acc, "topology": tm_accuracy},
            "sptm": {"type": tm_sp_type_acc, "topology": sptm_accuracy},
            "sp": {"type": sp_type_acc, "topology": sp_accuracy},
            "glob": {"type": glob_type_acc, "topology": glob_accuracy},
            "beta": {"type": beta_type_acc, "topology": beta_accuracy},
        },
    )
