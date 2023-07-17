# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    train_raw, valid_raw, test_raw, vocab = get_dataset_raw()
    lang = Lang(train_raw, ["<pad>", "<eos>"])

    train_dataset, valid_dataset, test_dataset = get_dataset(
        lang, train_raw, valid_raw, test_raw
    )
    train_loader, valid_loader, test_loader = get_dataloaders(
        lang, train_dataset, valid_dataset, test_dataset
    )

    criterion_train, criterion_eval = get_criterions(lang)
    # train(lang, train_loader, valid_loader, test_loader, criterion_train, criterion_eval, lstm=True, dropout=True, tie_weights=True, variational=True, ntasgd=True)

    eval(test_loader, criterion_eval, load_model(tie_weights=True))
    eval(test_loader, criterion_eval, load_model(variational=True))
    eval(test_loader, criterion_eval, load_model(ntasgd=True))