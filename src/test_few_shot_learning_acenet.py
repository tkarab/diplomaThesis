import test_NinaPro_Helper as npu
import constants
import numpy as np
from matplotlib import pyplot as plt

db=2
s=1
e=1
# Retrieve the data and format it in to a dataframe for each subject
for subj in range(1, 12):
    indata = npu.get_data(constants.getRaw_DB_Subject_path(db,s), constants.getRaw_subject_exercise_file_path(s,e))

    # The gesture repetitions to train on
    train_reps = [1, 3, 4, 6]

    # The gesture repetitions to test on
    test_reps = [2, 5]

    # Normalise the training data using scikit standardscaler
    data = npu.normalise(indata, train_reps)
    testdata = npu.normalise(indata, test_reps)

    # List for the 7 gesture ID's
    gestures = [i for i in range(1, 8)]

    # Use Windowing to extract and build the dataset
    # Set window length and stride in ms
    win_len = 200
    win_stride = 20

    # Build the train and test sets from the data with the set parameters
    X_train, y_train, r_train = npu.windowing(data, train_reps, gestures, win_len, win_stride)
    X_test, y_test, r_test = npu.windowing(testdata, test_reps, gestures, win_len, win_stride)

    print(y_train.shape)
    print(X_train.shape)

    # Make a few shot example set

    # Set the number of examples per class
    num_examples_per_class = 5

    # Get the unique classes in y_train
    classes = np.unique(y_train)

    # Initialize the mini datasets
    x_train_mini = []
    y_train_mini = []

    # Loop over each class
    for c in classes:
        # Get the indices of the examples for this class
        indices = np.where(y_train == c)[0]

        # Randomly select num_examples_per_class indices
        selected_indices = np.random.choice(indices, size=num_examples_per_class, replace=False)

        # Add the selected examples to the mini datasets
        x_train_mini.append(X_train[selected_indices])
        y_train_mini.append(y_train[selected_indices])

    # Convert the mini datasets to numpy arrays
    x_train_mini = np.concatenate(x_train_mini)
    y_train_mini = np.concatenate(y_train_mini)

    # Convert to one hot
    y_train = npu.get_categorical(y_train)
    y_test = npu.get_categorical(y_test)

    y_train_mini = npu.get_categorical(y_train_mini)