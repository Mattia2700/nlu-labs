# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
# Train and test with Stratified K Fold

from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
from transformers import BertTokenizer
import numpy as np
import nltk
import copy
import torch

from model import *
from utils import *

# nltk.download("subjectivity")
# from nltk.corpus import subjectivity

def lol2str(doc):
    # flatten & join
    return " ".join([w for sent in doc for w in sent])


def get_data():
    obj = subjectivity.sents(categories="obj")
    subj = subjectivity.sents(categories="subj")

    sentences = [lol2str(d) for d in obj] + [lol2str(d) for d in subj]
    ref = np.array([0] * len(obj) + [1] * len(subj))

    skf = KFold(n_splits=10, shuffle=True, random_state=42)
    dataset = Sentences(sentences, ref)

    return skf, dataset

def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.Linear]:
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias != None:
                m.bias.data.fill_(0.01)

def get_dataloaders(dataset, train_idx, test_idx):
    def collate_fn(batch):
        sentences, ref = zip(*batch)
        sentences = Parameters.TOKENIZER(
            list(sentences), padding=True, truncation=True, return_tensors="pt"
        )
        ref = torch.tensor(ref).float()
        sentences = sentences.to(Parameters.DEVICE)
        ref = ref.to(Parameters.DEVICE)
        sample = {"input_ids": sentences.input_ids, "attention_mask": sentences.attention_mask, "ref": ref}
        return sample

    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

    trainloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=Parameters.BATCH_SIZE,
        sampler=train_subsampler,
        collate_fn=collate_fn,
    )
    testloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=Parameters.BATCH_SIZE,
        sampler=test_subsampler,
        collate_fn=collate_fn,
    )

    print("Train size:", len(trainloader))
    print("Test size:", len(testloader))

    return trainloader, testloader


class Parameters:
    TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
    EPOCHS = 100
    BATCH_SIZE = 16
    TRAINING_CRITERION = nn.MSELoss()
    EVALUATION_CRITERION = nn.MSELoss()
    OPTIMIZER = torch.optim.Adam
    LR = 0.01
    PATIENCE = 3
    CLIP = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(skf, dataset):

    best_model_skf = None
    best_f1_skf = 0

    for i, (train_index, test_index) in enumerate(skf.split(dataset)):
        trainloader, testloader = get_dataloaders(dataset, train_index, test_index)

        model = Subjectivity()
        init_weights(model)
        model = model.to(Parameters.DEVICE)
        model.train()
        optimizer = Parameters.OPTIMIZER(model.parameters(), lr=Parameters.LR)

        losses_train = []
        losses_test = []
        best_model = None

        for x in tqdm(range(1, Parameters.EPOCHS+1)):
            # train loop
            
            train_loss = []
            test_loss = []

            for index, sample in enumerate(trainloader):
                optimizer.zero_grad()
                output = model(sample["input_ids"], sample["attention_mask"])
                loss = Parameters.TRAINING_CRITERION(output, sample["ref"])
                train_loss.append(loss.item())
                torch.nn.utils.clip_grad_norm_(model.parameters(), Parameters.CLIP)
                loss.backward()
                optimizer.step()

            if x % 5 == 0:
                losses_train.append(np.asarray(train_loss).mean())
                print("Train Loss:", train_loss[-1])

                ref = []
                pred = []

                # test loop
                model.eval()
                with torch.no_grad():
                    for sample in testloader:
                        output = model(sample["input_ids"], sample["attention_mask"])
                        output = [1 if o > 0.5 else 0 for o in output]
                        loss = Parameters.EVALUATION_CRITERION(output, sample["ref"])
                        test_loss.append(loss.item())
                        ref.extend(sample["ref"].tolist())
                        pred.extend(output.tolist())
                    
                losses_test.append(np.asarray(test_loss).mean())
                print("Test Loss:", losses_test[-1])

                rep = classification_report(ref, pred, zero_division=False, output_dict=True)
                f1 = rep["total"]["f"]
                print("F1:", f1)

                if f1 > best_f1:
                    best_f1 = f1
                    patience = Parameters.PATIENCE
                    best_model = copy.deepcopy(model)
                else:
                    patience -= 1
                if patience <= 0:  # Early stopping with patience
                    break  # Not nice but it keeps the code clean
                
            
        if best_f1 > best_f1_skf:
            best_f1_skf = best_f1
            best_model_skf = best_model

        print("Fold", i, "done")

    torch.save(best_model_skf.state_dict(), "model.pt")
