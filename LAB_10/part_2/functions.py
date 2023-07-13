# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
# Global variables
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Used to report errors on CUDA side

import numpy as np
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import copy

from model import *
from utils import *

try:
    from conll import evaluate
except ImportError:
    # downlaod it from https://raw.githubusercontent.com/BrownFortress/NLU-2023-Labs/main/labs/conll.py
    import requests

    url = "https://raw.githubusercontent.com/BrownFortress/NLU-2023-Labs/main/labs/conll.py"
    r = requests.get(url)
    with open("conll.py", "w") as f:
        if r.status_code == 200:
            f.write(r.text)
    from conll import evaluate


class Parameters:
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    PAD_TOKEN = 0

    HID_SIZE = 200
    EMB_SIZE = 300

    LR = 0.01  # learning rate
    CLIP = 5  # Clip the gradient

    OUT_SLOT = lambda x: len(x.slot2id)  # Number of output slot
    OUT_INT = lambda x: len(x.intent2id)  # Number of output intent
    VOCAB_LEN = lambda x: len(x.word2id)  # Vocabulary size

    CRITERSION_SLOTS = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    CRITERSION_INTENTS = nn.CrossEntropyLoss()  # Because we do not have the pad token

    N_EPOCHS = 200
    PATIENCE = 3


def get_dataset(train_raw, val_raw, test_raw):
    words = sum(
        [x["utterance"].split() for x in train_raw], []
    )  # No set() since we want to compute
    # the cutoff
    corpus = train_raw + val_raw + test_raw  # We do not wat unk labels,
    # however this depends on the research purpose
    slots = set(sum([line["slots"].split() for line in corpus], []))
    intents = set([line["intent"] for line in corpus])

    # Create our datasets
    train_dataset = IntentsAndSlots(train_raw)
    val_dataset = IntentsAndSlots(val_raw)
    test_dataset = IntentsAndSlots(test_raw)

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(train_dataset, val_dataset, test_dataset):
    # Dataloader instantiation
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)
    return train_loader, val_loader, test_loader


def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.xavier_uniform_(
                            param[idx * mul : (idx + 1) * mul]
                        )
                elif "weight_hh" in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.orthogonal_(param[idx * mul : (idx + 1) * mul])
                elif "bias" in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)


def train_loop(data, optimizer, criterion_slots, criterion_intents, model):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad()  # Zeroing the gradient
        slots, intent = model(
            sample["input_ids"], sample["attention_mask"], sample["token_type_ids"]
        )  # Forward pass
        loss_intent = criterion_intents(intent, sample["intent_label_ids"])
        # remove middle dimension from slots
        loss_slot = criterion_slots(slots, sample["slot_labels_ids"])
        loss = loss_intent + loss_slot  # In joint training we sum the losses.
        # Is there another way to do that?
        loss_array.append(loss.item())
        loss.backward()  # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()  # Update the weights
    return loss_array


def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad():  # It used to avoid the creation of computational graph
        for sample in data:
            slots, intents = model(sample["utterances"], sample["slots_len"])
            loss_intent = criterion_intents(intents, sample["intents"])
            loss_slot = criterion_slots(slots, sample["y_slots"])
            loss = loss_intent + loss_slot
            loss_array.append(loss.item())
            # Intent inference
            # Get the highest probable class
            out_intents = [
                lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()
            ]
            gt_intents = [lang.id2intent[x] for x in sample["intents"].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # Slot inference
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample["slots_len"].tolist()[id_seq]
                utt_ids = sample["utterance"][id_seq][:length].tolist()
                gt_ids = sample["y_slots"][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append(
                    [(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)]
                )
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        pass

    report_intent = classification_report(
        ref_intents, hyp_intents, zero_division=False, output_dict=True
    )
    return results, report_intent, loss_array


def train(
    train_loader, val_loader, test_loader, bidirectional=False, dropout=None
):
    model = ModelIAS(
        50,
        50,
        # Parameters.HID_SIZE,
        # Parameters.OUT_SLOT(lang),
        # Parameters.OUT_INT(lang),
        # Parameters.EMB_SIZE,
        # Parameters.VOCAB_LEN(lang),
        # pad_index=Parameters.PAD_TOKEN,
        # bidirectional=bidirectional,
        # dropout=dropout,
    ).to(Parameters.DEVICE)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=Parameters.LR)

    n_epochs = Parameters.N_EPOCHS
    patience = Parameters.PATIENCE
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0
    best_model = None

    for x in tqdm(range(1, n_epochs)):
        loss = train_loop(
            train_loader,
            optimizer,
            Parameters.CRITERSION_SLOTS,
            Parameters.CRITERSION_INTENTS,
            model,
        )
        if x % 5 == 0:
            sampled_epochs.append(x)
            losses_train.append(np.asarray(loss).mean())
            results_dev, intent_res, loss_dev = eval_loop(
                val_loader,
                Parameters.CRITERSION_SLOTS,
                Parameters.CRITERSION_INTENTS,
                model,
            )
            losses_dev.append(np.asarray(loss_dev).mean())
            f1 = results_dev["total"]["f"]

            if f1 > best_f1:
                best_f1 = f1
                patience = Parameters.PATIENCE
                best_model = copy.deepcopy(model)
            else:
                patience -= 1
            if patience <= 0:  # Early stopping with patience
                break  # Not nice but it keeps the code clean

    results_test, intent_test, _ = eval_loop(
        test_loader,
        Parameters.CRITERSION_SLOTS,
        Parameters.CRITERSION_INTENTS,
        model,
        lang,
    )
    print("Slot F1: ", results_test["total"]["f"])
    print("Intent Accuracy:", intent_test["accuracy"])
    torch.save(best_model, "bin/best_model.pt")
