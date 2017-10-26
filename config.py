class Config:
    """
    Contains all the important configurations.
    """
    # Frequency refers to the size of each data set (ensure that the number used is divisible by 10).
    frequency = 10000
    # Test size refers to the percentage of the data set allocated to the test set.
    test_size = 0.20
    # Lexicon size refers to the maximum number of words in the lexicon.
    lexicon_size = 1000
    # Batch size refers to the number of features in each batch
    batch_size = 1024
