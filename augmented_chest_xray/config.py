# Dataset information
LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]
USE_COLUMNS = ['Image Index', 'Finding Labels', 'Patient ID']
N_CLASSES = len(LABELS)


# Preprocessing
# - default arguments
RANDOM_SEED = 42
DATASET_IMAGE_SIZE = 256
TRAIN_VAL_TEST_RATIO = (.7, .1, .2)

# - input filenames and directories
IMAGE_DIR = './data/original_data/images/'
META_DATA_DIR = './data/original_data/meta_data/'
META_FILENAME = 'Data_Entry_2017_v2020.csv'

# - output filenames and directories
LABELS_DIR = './data/labels/'
DATASET_DIR = './data/datasets/'
TRAIN_LABELS = 'train_labels.csv'
VAL_LABELS = 'val_labels.csv'
TEST_LABELS = 'test_labels.csv'


# Training, validation and testing
# - filenames and directories
SAVE_DIR = './data/saved_models/'

# - default arguments
BATCH_SIZE = 8
NUM_WORKERS = 1
NUM_EPOCHS = 8
LEARNING_RATE = 0.001
SCHEDULER_FACTOR = 0.1
SCHEDULER_PATIENCE = 0
DATA_AUGMENTATION = False
CENTER_CROP = DATA_AUGMENTATION
DEBUG = False


# Testing and debugging
DEBUG_SIZES = {'train': 1280, 'val': 1280, 'test':1280}
