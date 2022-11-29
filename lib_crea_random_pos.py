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


def place_kings(brd):
	while True:
		rank_white, file_white, rank_black, file_black = random.randint(0,7), random.randint(0,7), random.randint(0,7), random.randint(0,7)
		diff_list = [abs(rank_white - rank_black),  abs(file_white - file_black)]
		if sum(diff_list) > 2 or set(diff_list) == set([0, 2]):
			brd[rank_white][file_white], brd[rank_black][file_black] = "K", "k"
			break
 
def populate_board(brd, wp, bp, piece_list):
	for x in range(2):
		if x == 0:
			piece_amount = wp
			pieces = piece_list
		else:
			piece_amount = bp
			pieces = [s.lower() for s in piece_list]
		while piece_amount != 0:
			piece_rank, piece_file = random.randint(0, 7), random.randint(0, 7)
			piece = random.choice(pieces)
			if brd[piece_rank][piece_file] == " " and pawn_on_promotion_square(piece, piece_rank) == False:
				brd[piece_rank][piece_file] = piece
				piece_amount -= 1
 
def fen_from_board(brd):
	fen = ""
	for x in brd:
		n = 0
		for y in x:
			if y == " ":
				n += 1
			else:
				if n != 0:
					fen += str(n)
				fen += y
				n = 0
		if n != 0:
			fen += str(n)
		fen += "/" if fen.count("/") < 7 else ""
	fen += " w - - 0 1\n"
	return fen
 
def pawn_on_promotion_square(pc, pr):
	if pc == "P" and (pr == 0 or pr == 7):
		return True
	elif pc == "p" and (pr == 0 or pr == 7):
		return True
	return False
 
 
def start_random_position(n_pieces):
	board_ = [[" " for x in range(8)] for y in range(8)]
	piece_list = ["R", "N", "B", "Q", "P"]     
	piece_amount_white = random.randint(1, n_pieces-2-1) #ogni colore almeno un pezzo in più oltre al re
	piece_amount_black = n_pieces-2-piece_amount_white
	place_kings(board_)
	populate_board(board_, piece_amount_white, piece_amount_black, piece_list)
	#print(fen_from_board(board))
	#for x in board:
	#	print(x)
	return fen_from_board(board_)
    
#entry point
#start_random_position()

def get_valid_random_fens(min_pieces, max_pieces, fen_for_pieces):
    random_fens = []
    for n_pieces in range(min_pieces, max_pieces+1):
        for n_pos in range(0,fen_for_pieces):
            is_valid = False
            while not is_valid:
                fen = start_random_position(n_pieces)
                engine = chess.engine.SimpleEngine.popen_uci(stockfishpath)
                board = chess.Board(fen)
                is_not_valid = not board.is_valid()
                board = chess.Board(fen.replace(' b ',' w '))
                is_check_b = board.is_check()
                board = chess.Board(fen.replace(' w ',' b '))
                is_check_w = board.is_check()
                is_valid = not (is_check_b or is_check_w or is_not_valid)
                engine.quit()
            random_fens.append([fen,n_pieces])
            
    valid_random_fens = pd.DataFrame(random_fens, columns=['fen','n_pieces'])
    return valid_random_fens
    
