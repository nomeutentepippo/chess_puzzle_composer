import pandas as pd
import math
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
import numpy as np
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 100)
from os.path import exists
from stockfish import Stockfish
import chess
import chess.engine
import chess.pgn
import random
import time
from joblib import Parallel, delayed
import multiprocessing
import pickle
import os
import logging
import func_timeout
stockfishpath=r"C:\Program Files\stockfish_15_win_x64_avx2\stockfish_15_x64_avx2.exe"
from libreria_analisi import analyze_all_fens
from lib_crea_random_pos import get_valid_random_fens
from  lib_load_fen_data import load_m8_list, parse_png


valid_random_fens_1 = get_valid_random_fens(11,12,50)
print(valid_random_fens_1.shape)
all_evaluation_random_pos_1, total_time = analyze_all_fens(valid_random_fens_1['fen'], problem_type=0, multi_thread=False)
print(f'random_pos: {total_time}')      
all_evaluation_random_pos_1.to_pickle(r"C:\Users\Giorgio\Documents\Python Scripts\Scacchi puzzle\chess_puzzles\all_evaluation_random_pos_11-12_man.pkl")