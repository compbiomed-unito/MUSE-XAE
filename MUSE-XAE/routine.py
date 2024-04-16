import pandas as pd
import numpy as np
from de_novo import denovo_extraction
from assignment import signature_assignment
import os
import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation


def main_process(args):

    if args.refit_only=='False':
        denovo_extraction(args)
    else:
        signature_assignment(args)



