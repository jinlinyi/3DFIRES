from yacs.config import CfgNode as CN


def get_cfg_defaults():
    """
    Customize the detectron2 cfg to include some new keys and default values
    """
    cfg = CN()
    cfg.VIS_PERIOD = 200

    cfg.MODEL = CN()
    cfg.MODEL.BACKBONE = CN()
    cfg.MODEL.BACKBONE.NAME = "dpt_beit_large_384"
    cfg.MODEL.FREEZE = []

    cfg.MODEL.DRDF_TAU = 1.0
    cfg.MODEL.DEPTH_ONLY = False
    cfg.MODEL.DECAY_LOSS_ON = False

    cfg.MODEL.DRDF_MLP = CN()
    cfg.MODEL.DRDF_MLP.MLP_BATCH_NORM = False
    cfg.MODEL.DRDF_MLP.MLP_FEATURE_DIM = 256
    cfg.MODEL.DRDF_MLP.RAY_ATTENTION_ON = False
    cfg.MODEL.DRDF_MLP.CAM_ATTENTION_ON = False
    cfg.MODEL.DRDF_MLP.POSITIONAL_ENCODING_FREQ = 6

    cfg.DATASETS = CN()
    cfg.DATASETS.TRAIN = ["train_set3"]
    cfg.DATASETS.TEST = ["val_set3"]

    cfg.DATASET_GENERATE = CN()
    cfg.DATASET_GENERATE.RAY_SAMPLE_RESOLUTION = 0.025
    cfg.DATASET_GENERATE.ZNEAR = 0.1
    cfg.DATASET_GENERATE.ZFAR = 8.0
    cfg.DATASET_GENERATE.MAX_HIT = 8
    cfg.DATASET_GENERATE.FIX_FOV_DEG = None

    cfg.DATALOADER = CN()
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.DATALOADER.NUM_RAY_PER_IMG_SAMPLE = 200
    cfg.DATALOADER.RESIZE = 128
    cfg.DATALOADER.SAMPLE_GAUSSIAN_ON = False
    cfg.DATALOADER.GAUSSIAN_STD = 0.4
    cfg.DATALOADER.NUM_GAUSSIAN_PT = 256  # only when SAMPLE_GAUSSIAN_ON is True
    cfg.DATALOADER.NUM_UNIFORM_PT = 256  # only when SAMPLE_GAUSSIAN_ON is True
    cfg.DATALOADER.ADAPTIVE_SAMPLING = False
    cfg.DATALOADER.CAMERA_MODE = "perspective"
    cfg.DATALOADER.TRAIN_VIEW = []
    cfg.DATALOADER.TEST_VIEW = []
    cfg.DATALOADER.HIGH_RES_OUTPUT = False

    cfg.DATALOADER.CACHE_PATH = ""
    cfg.DATALOADER.TAR_ROOT = ""
    cfg.DATALOADER.INDEX_ROOT = ""
    cfg.EVAL_PERIOD = 1000
    cfg.SAVE_PERIOD = 1000

    cfg.TRAIN = CN()
    cfg.TRAIN.EPOCHS = 80
    cfg.TRAIN.SOLVER = CN()

    cfg.TRAIN.SOLVER.TYPE = "SGD"
    cfg.TRAIN.SOLVER.MOMENTUM = 0.9
    cfg.TRAIN.SOLVER.BASE_LR = 0.001
    cfg.TRAIN.SOLVER.LR_SCHEDULER_NAME = "StepLR"
    cfg.TRAIN.SOLVER.GAMMA = 0.1
    cfg.TRAIN.SOLVER.STEPS = 26
    cfg.TRAIN.SOLVER.WEIGHT_DECAY = 0.0001
    return cfg
