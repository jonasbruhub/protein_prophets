import torch
from torch.utils.data import Dataset
import pandas as pd

import os
import json
from tqdm import tqdm

from utils.metrics_utils import LABELS, LABELS_INV


def GetAlphaFoldProteinsUsed():
    path = "Data/AlphaFoldDBEncoded/"

    AlphaFoldResults = [
        f[:-3]
        for f in os.listdir(path)
        if (os.path.isfile(path + "/" + f) & f.__contains__(".pt"))
    ]

    # The following 3 proteins did not match between AlphaFold and the supplied JSON. Running the code to find these took a very long time.
    BlackListProteins = ["Q841A2", "D6R8X8", "Q8I2A6"]

    # These were in the json with data trained on, but not in crossval results (https://biolib-public-assets.s3.eu-west-1.amazonaws.com/deeptmhmm/DeepTMHMM.crossval.top)
    BlackListProteins += ["P02930", "A1JUB7"]

    proteinIDs = []
    AlphaFoldResultsSet = set(AlphaFoldResults)

    with open("Data/DeepTMHMM.partitions.json", "r") as FileObj:
        CVs = json.loads(FileObj.read())
        for cv in CVs.keys():
            cvProteins = CVs[cv]
            for idx, protein in enumerate(cvProteins):
                if protein["sequence"].__len__() > 1_500:
                    continue
                if protein["id"] in BlackListProteins:
                    continue
                if protein["id"] in AlphaFoldResultsSet:
                    proteinIDs += [
                        protein["id"],
                    ]

    return proteinIDs


def GetProteinMap(rel_path=""):
    path = rel_path + "Data/AlphaFoldDBEncoded/"

    AlphaFoldResults = [
        f[:-3]
        for f in os.listdir(path)
        if (os.path.isfile(path + "/" + f) & f.__contains__(".pt"))
    ]

    # The following 3 proteins did not match between AlphaFold and the supplied JSON. Running the code to find these took a very long time.
    BlackListProteins = ["Q841A2", "D6R8X8", "Q8I2A6"]

    # These were in the json with data trained on, but not in crossval results (https://biolib-public-assets.s3.eu-west-1.amazonaws.com/deeptmhmm/DeepTMHMM.crossval.top)
    BlackListProteins += ["P02930", "A1JUB7"]

    proteinIDs = []
    AlphaFoldResultsSet = set(AlphaFoldResults)

    # Get proteins where:
    #   Is in AlphaFold
    #   Is atmost 1500 amino acids long
    #   Is not in blacklist (AlphaFold and JSON sequence does not match)
    with open(rel_path + "Data/DeepTMHMM.partitions.json", "r") as FileObj:
        CVs = json.loads(FileObj.read())
        for cv in CVs.keys():
            cvProteins = CVs[cv]
            for idx, protein in enumerate(cvProteins):
                if protein["sequence"].__len__() > 1_500:
                    continue
                if protein["id"] in BlackListProteins:
                    continue
                if protein["id"] in AlphaFoldResultsSet:
                    proteinIDs += [
                        [
                            protein["id"],
                            protein,
                            protein["sequence"],
                            protein["labels"],
                            cv,
                            idx,
                        ]
                    ]

    columns = ["proteinID", "protein", "sequence", "labels", "CV", "index"]
    proteinMap = pd.DataFrame(proteinIDs, columns=columns)
    proteinMap.index = proteinMap["proteinID"].values

    return proteinMap

def GetLSTMData():
    # should return a dict {cv: list of [latent, label] }
    proteinMap = GetProteinMap()
    cvs = proteinMap["CV"].unique()
    print(cvs)
    path = "Data/AlphaFoldDBEncoded/"

    foldDict = {}
    for cv in cvs:
        foldDict[cv] = []
        print(f"preparing {cv[-1]} fold")
        cvProteins = proteinMap[proteinMap["CV"] == cv]
        for _, protein in tqdm(cvProteins.iterrows()):
            
            # Latent shoud be of size Lx512 from encoder. Is by default when loading precomputed
            latent = torch.load(
                    path + "/" + protein["proteinID"] + ".pt"
                )
            
            # Label should be a list of characters = [S,S,S,S,P,P,B,B,...]
            label = list(protein["labels"])
            foldDict[cv] += [[latent.unsqueeze(1), label]]
    return foldDict


def GetCustomProteinDataset(encode_length=1500):
    proteinMap = GetProteinMap()

    path = "Data/AlphaFoldDBEncoded/"

    class CustomProteinDataset(Dataset):
        def __init__(self, protein_code, transform=None, target_transform=None):
            self.proteins = proteinMap.loc[protein_code][["proteinID", "labels"]]
            self.transform = transform
            self.target_transform = target_transform
            self.proteinsEncoded = []
            self.labels = []
            print("encoding proteins")
            for index, protein in tqdm(self.proteins.iterrows()):
                latent = torch.load(
                    path + "/" + protein["proteinID"] + ".pt"
                )  # shape = [length, 512]
                self.proteinsEncoded += [
                    torch.cat(
                        [
                            latent,
                            torch.zeros(
                                (encode_length - latent.shape[0], latent.shape[1])
                            ),
                        ],
                        0,
                    )
                ]
                self.labels += [
                    torch.tensor(
                        EncodeLabel(
                            protein["labels"].ljust(encode_length, LABELS_INV[-1])
                        )
                    )
                ]
            self.proteinsEncoded = torch.stack(self.proteinsEncoded, 0).permute(0, 2, 1)
            self.labels = torch.stack(self.labels, 0)

        def __len__(self):
            return self.proteinsEncoded.shape[0]

        def __getitem__(self, idx):
            encodeLatent = self.proteinsEncoded[idx]
            label = self.labels[idx]
            if self.transform:
                encodeLatent = self.transform(encodeLatent)
            if self.target_transform:
                label = self.target_transform(label)
            return encodeLatent, label

    return CustomProteinDataset


def GetCustomProteinDatasetPadded(encode_length=1500):
    proteinMap = GetProteinMap()

    path = "Data/AlphaFoldDBEncoded/"

    class CustomProteinDataset(Dataset):
        def __init__(self, protein_code, transform=None, target_transform=None):
            self.proteins = proteinMap.loc[protein_code][["proteinID", "labels"]]
            self.transform = transform
            self.target_transform = target_transform
            self.proteinsEncoded = []
            self.labels = []
            print("encoding proteins")
            for index, protein in tqdm(self.proteins.iterrows()):
                latent = torch.load(path + "/" + protein["proteinID"] + ".pt")
                paddedLatent = torch.cat(
                    [
                        latent,
                        torch.ones((latent.shape[0], 1)),
                    ],
                    1,
                )
                # self.proteinsEncoded += []
                self.proteinsEncoded += [
                    torch.cat(
                        [
                            paddedLatent,
                            torch.zeros(
                                (encode_length - latent.shape[0], paddedLatent.shape[1])
                            ),
                        ],
                        0,
                    )
                ]
                self.labels += [
                    torch.tensor(
                        EncodeLabel(
                            protein["labels"].ljust(encode_length, LABELS_INV[-1])
                        )
                    )
                ]
            self.proteinsEncoded = torch.stack(self.proteinsEncoded, 0).permute(0, 2, 1)
            self.labels = torch.stack(self.labels, 0)

        def __len__(self):
            return self.proteinsEncoded.shape[0]

        def __getitem__(self, idx):
            encodeLatent = self.proteinsEncoded[idx]
            label = self.labels[idx]
            if self.transform:
                encodeLatent = self.transform(encodeLatent)
            if self.target_transform:
                label = self.target_transform(label)
            return encodeLatent, label

    return CustomProteinDataset


def EncodeLabel(label):
    return [LABELS[lab] for lab in label]


def GetCVProteins():
    proteinMap = GetProteinMap()

    with open("Data/DeepTMHMM.partitions.json", "r") as FileObj:
        CVs = json.loads(FileObj.read())
        cvList = list(CVs.keys())

    CVProteins = {}
    for cv in cvList:
        CVProteins[cv] = list(proteinMap[proteinMap.CV == cv].index.values)

    return CVProteins
