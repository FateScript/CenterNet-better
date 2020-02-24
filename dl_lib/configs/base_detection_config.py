from .base_config import BaseConfig

_config_dict = dict(
    MODEL=dict(
        LOAD_PROPOSALS=False,
        MASK_ON=False,
        KEYPOINT_ON=False,
        RESNETS=dict(
            DEPTH=50,
            OUT_FEATURES=["res4"],
            NUM_GROUPS=1,
            NORM="FrozenBN",
            WIDTH_PER_GROUP=64,
            STRIDE_IN_1X1=True,
            RES5_DILATION=1,
            RES2_OUT_CHANNELS=256,
            STEM_OUT_CHANNELS=64,
        ),
        PROPOSAL_GENERATOR=dict(MIN_SIZE=0,),
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=(800,), max_size=1333, sample_style="choice")),
                ("RandomFlip", dict()),
            ],
            TEST_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=800, max_size=1333, sample_style="choice")),
            ],
        ),
        CROP=dict(ENABLED=False, TYPE="relative_range", SIZE=[0.9, 0.9],),
        MASK_FORMAT="polygon",
    ),
    DATASETS=dict(
        TRAIN=(),
        PROPOSAL_FILES_TRAIN=(),
        PRECOMPUTED_PROPOSAL_TOPK_TRAIN=2000,
        TEST=(),
        PROPOSAL_FILES_TEST=(),
        PRECOMPUTED_PROPOSAL_TOPK_TEST=1000,
    ),
    TEST=dict(
        EXPECTED_RESULTS=[],
        EVAL_PERIOD=0,
        KEYPOINT_OKS_SIGMAS=[],
        DETECTIONS_PER_IMAGE=100,
        AUG=dict(
            ENABLED=False,
            MIN_SIZES=(400, 500, 600, 700, 800, 900, 1000, 1100, 1200),
            MAX_SIZE=4000,
            FLIP=True,
        ),
        PRECISE_BN=dict(ENABLED=False, NUM_ITER=200),
    ),
)


class BaseDetectionConfig(BaseConfig):
    def __init__(self):
        super(BaseDetectionConfig, self).__init__()
        self._register_configuration(_config_dict)


config = BaseDetectionConfig()
