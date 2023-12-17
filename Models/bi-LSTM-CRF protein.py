# Inspiration from pytorch tutorials
# https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html?fbclid=IwAR2NNcwPuDh57KxrZXU9BYT5RiVpX9nG2pEE18WyKStCf4q1iPHoV8thg3k

import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from utils.setup import GetLSTMData
from utils.metrics_utils import confusionMatrix
from tqdm import tqdm

# Get all data for training etc.
cvs = GetLSTMData()

training_data = cvs["cv0"] + cvs["cv1"] + cvs["cv2"]
validation_data = cvs["cv3"]
test_data = cvs["cv4"]



# Only take a subset of data
# training_data = training_data[:2]
# validation_data = validation_data[:2]



batch_size = 32
validation_every_steps = 50
num_epochs = 300

print_error_type_pairs = False

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("using device:", device)

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BiLSTM_CRF(nn.Module):
    def __init__(self, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        # self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size).to(device))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2).to(device),
                torch.randn(2, 1, self.hidden_dim // 2).to(device))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.).to(device)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1).to(device)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()

        # embeds here should be the given latent variables
        # embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        
        embeds = sentence

        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).to(device)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).to(device), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.).to(device)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t).to(device) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        # print(feats)
        # feats is a Lxhidden_dim vector
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # print(lstm_feats.shape)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 512
HIDDEN_DIM = 50

tag_to_ix = {"I": 0, "O": 1, "P": 2, "S": 3, "M": 4, "B": 5, START_TAG: 6, STOP_TAG: 7}
tag_to_ix_inv = {0: "I", 1:"O", 2:"P", 3:"S", 4:"M", 5:"B", 6:START_TAG, 7:STOP_TAG}

model = BiLSTM_CRF(tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM).to(device)

# # Check predictions before training
# with torch.no_grad():
#     print("Prediction before training")
#     model_pred = model(training_data[0][0].to(device))
#     pred_label = ''.join([tag_to_ix_inv[w] for w in model_pred[1]])
#     print( f"  predict: { pred_label }")
#     print( f"  target:  {''.join(training_data[0][1])}")


# Adam or SGD
# optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)





step = 0
valid_accuracies = []
loss_total = 0



# Set model to train
model.train()

# Batch size no more than number of training points
batch_size = min(batch_size,training_data.__len__())

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(num_epochs):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Increment step counter
        step += 1
        sentence = sentence.to(device)

        # Step 1. Clear gradients
        model.zero_grad()

        # Step 2. Convert target labels to intigers
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long).to(device)

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence, targets)

        # Update accumulated loss
        loss_total += loss/batch_size


        # Step 4. Compute the loss, gradients, and update the parameters by calling optimizer.step()
        # loss.backward()
        # optimizer.step()
        if step % batch_size == 0:
            # print("update!!")
            # print(loss_total)
            loss_total.backward()
            optimizer.step()
            loss_total = 0
        

        # Print loss
        # if (epoch % 10 == 0):
        #     print(loss)
        
        if step % validation_every_steps == 0:
            validation_accuracies_batches = []
            prediction_labels_list = []
            target_labels_list = []

            with torch.no_grad():
                model.eval()
                for sentence, tags in validation_data:
                    sentence = sentence.to(device)

                    # Get indices of amino acid prediction labels
                    output = model(sentence)
                    # print(output)

                    prediction_labels_list += [output[1]]
                    target_labels_list += [[tag_to_ix[w] for w in tags]]
                
            model.train()

            (
                error_type_pairs,
                confusion_matrix,
                type_accuracy,
                detailed_type_accuracy,
            ) = confusionMatrix(prediction_labels_list, target_labels_list)

            print("\n\nValidation")
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









print("\n\n\nTESTING  MODEL")

# Now we test the model on the test set
with torch.no_grad():
    model.eval()
    for sentence, tags in test_data:
        sentence = sentence.to(device)


        # inputs, targets = batch
        # inputs, tragets = inputs.to(device), targets.to(device)
        # output = model(inputs[:, :-1, :])
        output = model(sentence)

        predictions = output[1]

        # for idx in range(predictions.shape[0]):
        # target_len = int(torch.sum(inputs[idx, -1, :]))
        # test_accuracies.append(
        #     accuracy(
        #         targets[idx][0:target_len],
        #         predictions[idx][0:target_len],
        #     )
        # )
        prediction_labels_list += [output[1]]
        # target_labels_list += [targets[idx][0:target_len]]
        target_labels_list += [[tag_to_ix[w] for w in tags]]

# print("  Test accuracy: " + str(np.sum(test_accuracies) / len(test_dataset)))

# Extra accuracies
(
    error_type_pairs,
    confusion_matrix,
    type_accuracy,
    detailed_type_accuracy,
) = confusionMatrix(prediction_labels_list, target_labels_list)

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














# print("\n\n\nA sanity check on first training input. Prediction and target")
# # Check predictions after training
# with torch.no_grad():
#     # Input should just be latent structure
#     print("Prediction after training")
#     model_pred = model(training_data[0][0].to(device))
#     pred_label = ''.join([tag_to_ix_inv[w] for w in model_pred[1]])
#     print( f"  predict: { pred_label }")
#     print( f"  target:  {''.join(training_data[0][1])}")

# # We got it!
