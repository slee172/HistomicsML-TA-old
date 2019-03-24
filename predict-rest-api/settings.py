class Settings():

    def __init__(self):

        # initialize redis settings
        self.REDIS_HOST = "localhost"
        self.REDIS_PORT = 6379
        self.REDIS_DB = 0
        self.MYSQL_HOST = "172.17.0.5"

        # initialize constants
        self.REQUEST_QUEUE = "request_queue"
        self.REQUEST_START = 0
        self.REQUEST_END = 100
        self.SLEEP = 0.5

        # initialize datasets
        self.FEATURE_DIM = 64
        self.IS_HEATMAP = False
        self.TRAININGSET_DIR = "/localdata/classifiers/"
        self.DATASET_DIR = "/fastdata/features/"
        self.PATH_TO_SPECIAL = "/fastdata/features/BRCA/BRCA-spfeatures-46-24-01.h5"
      # self.PATH_TO_SPECIAL = "/fastdata/features/SKCM/SKCM-spfeatures-48-11-01.h5"
