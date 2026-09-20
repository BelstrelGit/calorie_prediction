class Config:
    SEED = 42
    DATA_DIR = "data"
    MODEL_PATH = "models/best_model.pt"

    TEXT_MODEL_NAME = "bert-base-uncased"
    IMAGE_MODEL_NAME = "tf_efficientnet_b0"

    IMAGE_SIZE = 224
    MAX_LENGTH = 128

    BATCH_SIZE = 4
    VAL_SIZE = 0.15

    TEXT_LR = 3e-5
    IMAGE_LR = 1e-4
    HEAD_LR = 1e-3

    NUM_EPOCHS = 5

    DEBUG = False
    DEBUG_TRAIN_SIZE = 32
    DEBUG_VAL_SIZE = 16