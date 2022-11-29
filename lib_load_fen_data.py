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

def load_m8_list(filename, n_skip_lines):
    df = pd.read_csv(filename, skiprows=n_skip_lines, header=None, sep=';')
    df = df[~df[0].str.startswith('White Mates in')]
    df = df[~df[0].str.startswith('Black Mates in')]
    df1=pd.concat((df.iloc[0::3].reset_index(drop=True), df.iloc[1::3].reset_index(drop=True),df.iloc[2::3].reset_index(drop=True)), axis=1)
    df1.columns = ['info','fen','solution']
    df1
    return df1
    
def parse_png(png_file_name):
    pgns_parsed=[]
    pgn = open(png_file_name, encoding="utf-8", errors='ignore')
    while True:
        game=chess.pgn.read_game(pgn)
        if game is None:
            break        
        #var = []
        #for node in game.mainline_moves():
        #    var.append(node.move)
        if ('FEN' in game.headers) and ('Result' in game.headers):
            pgns_parsed.append((game.headers, game.headers['FEN'], game.mainline_moves(), game.headers['Result']))
    pgns_parsed=pd.DataFrame(pgns_parsed, columns=['info', 'fen', 'solution','result'])
    pgn.close()
    return pgns_parsed