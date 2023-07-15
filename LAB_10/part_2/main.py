# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    # Wrtite the code to load the datasets and to run your functions
    # Print the results
    tmp_train_raw, test_raw = load_dataset()
    train_raw, val_raw, test_raw = split_dataset(tmp_train_raw, test_raw)

    train_dataset, val_dataset, test_dataset, lang = get_dataset(
        train_raw, val_raw, test_raw
    )
    train_loader, val_loader, test_loader = get_dataloaders(
        train_dataset, val_dataset, test_dataset
    )

    # train(train_loader, val_loader, test_loader, lang)
    eval(test_loader, load_model(), lang)
