class Config(object):
    def __init__(self):

        # Learning Rates
        self.lr = 5e-4

        # Epochs
        self.epochs = 10
        self.batch_size = 8

        # Basic
        self.n_worker = 4

        # Model
        self.model = 'FPN'
        self.encoder = 'efficientnet-b7'
        self.pre_trained_weight = 'imagenet'

