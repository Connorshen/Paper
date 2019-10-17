from experiment.trainer.cnn_bp_trainer import CnnBpTrainer
import numpy as np

batch_size = 40
digits = np.array([3, 5])
epoch = 1
use_gpu = True
cnn_bp_trainer = CnnBpTrainer(batch_size,
                              digits,
                              epoch,
                              use_gpu)
cnn_bp_trainer.run_training()
