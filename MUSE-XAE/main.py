import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation
from input_parser import parser_args
from routine import main_process
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='keras.engine.training_v1')

tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


if __name__ == '__main__':
    
    args=parser_args()
    main_process(args)




