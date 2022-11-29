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


def has_one_and_only_one_solution(depth_eval, depth_level):
    idx = pd.IndexSlice
    return (depth_eval.loc[idx[:], idx[depth_level,'n_best', 1]] == 1) & (depth_eval.loc[idx[:], idx[depth_level,'n_best', :]].sum(axis=1) < 2)
def top_level_is_correct(depth_eval, depth_level):
    idx = pd.IndexSlice
    return depth_eval.loc[idx[:], idx[depth_level,'different_levels', 20]] == 1
def not_all_levels_are_correct(depth_eval, depth_level):
    idx = pd.IndexSlice
    return depth_eval.loc[idx[:], idx[depth_level,'different_levels', :]].sum(axis=1) < 3
def result_from_eval(evaluation):
    ev = int(evaluation)
    if abs(ev) < 50:
        return 0
    elif ev > 300:
        return 1
    elif ev < -300:
        return -1
    else:
        return None
        
def result_from_eval_array(evaluation):
    result = evaluation.copy()
    #result.loc[:,:] = None
    #result.loc[abs(evaluation) < 0.5] = 0
    #result.loc[evaluation > 400] = 1
    #result.loc[evaluation < -400] = -1
    for i in range(0,len(evaluation)):
        if i < len(evaluation)-3:
            ev = int(evaluation[i])
            result[i] = result_from_eval(ev)
        else:
            result[i] = None
    return result
    
def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def elaborate_evaluation_for_problem(all_evaluation):
    idx = pd.IndexSlice
    move_of_problem = all_evaluation.loc[idx[:], idx[:,:, :, ['move',''],:]]
    move_of_problem = move_of_problem.droplevel(['score_move'], axis=1)

    eval_of_problem = all_evaluation.loc[idx[:], idx[:,:, :, ['centipawn',''],:]]
    eval_of_problem = eval_of_problem.droplevel(['score_move'], axis=1)
    #eval_of_problem = eval_of_problem.drop(columns=['fen', 'total_time', 'problem_type'])
    eval_of_problem = eval_of_problem.fillna(value=100)

    correct_move_of_problem = move_of_problem.copy()
    for i in range(0,correct_move_of_problem.shape[0]): 
        #result = result_from_eval((eval_of_problem.loc[i,('depth_25', 'n_best', 1)] + eval_of_problem.loc[i,('depth_25', 'different_levels', 20)])/2)
        result = result_from_eval(eval_of_problem.loc[i,('depth_25', 'different_levels', 20)])
        result_moves = correct_move_of_problem.loc[i, result_from_eval_array(eval_of_problem.loc[i,])==result].dropna().unique()
        result_moves = intersection(result_moves, move_of_problem.loc[i,('depth_25', slice(None))].tolist())
        #result_moves = [move_of_problem.loc[i,('depth_25', 'different_levels', 20)]]
        #tengo solo le mosse uguali alla mossa migliore se la mossa migliore è matto
        if result_from_eval(eval_of_problem.loc[i,('depth_25','different_levels', 20)]) is not None: #la mossa migliore da un risultato
            correct_move_of_problem.loc[i,correct_move_of_problem.loc[i,:].isin(result_moves) == False] = np.nan
        else:
            correct_move_of_problem.loc[i,:]=np.nan
    correct_move_of_problem.drop(columns=['fen','total_time','problem_type'], inplace=True)
    solution_map = correct_move_of_problem.notnull().astype('int')

    acceptable_solution = solution_map[(solution_map.iloc[:,-1]==solution_map.iloc[:,-4]) & (solution_map.iloc[:,-1]==1)]
    regret_solution = solution_map[~((solution_map.iloc[:,-1]==solution_map.iloc[:,-4]) & (solution_map.iloc[:,-1]==1))]
    
    return solution_map, acceptable_solution, regret_solution
    
def elaborate_evaluation_for_mate(all_evaluation):
    #n_move_mate=mate_in    
    idx = pd.IndexSlice
    move_of_mate = all_evaluation.loc[idx[:], idx[:,:, :, ['move',''],:]]
    move_of_mate = move_of_mate.droplevel(['score_move'], axis=1)

    eval_of_mate = all_evaluation.loc[idx[:], idx[:,:, :, ['mate',''],:]]
    eval_of_mate = eval_of_mate.droplevel(['score_move'], axis=1)
    eval_of_mate = eval_of_mate.fillna(value=1000)
    
    eval_of_centipawn = all_evaluation.loc[idx[:], idx[:,:, :, ['centipawn',''],:]]
    eval_of_centipawn = eval_of_centipawn.droplevel(['score_move'], axis=1)
    #eval_of_centipawn = eval_of_centipawn.fillna(value=100)

    #correct_move_of_mate = move_of_mate[eval_of_mate==n_move_mate]
    #for i in range(0,correct_move_of_mate.shape[0]): #tengo solo le mosse uguali alla mossa migliore
    #    correct_move_of_mate.loc[i,correct_move_of_mate.loc[i,:]!=correct_move_of_mate.loc[i,('depth_25','different_levels', 20)]]=np.nan
    correct_move_of_mate=move_of_mate.copy()
    for i in range(0,correct_move_of_mate.shape[0]): 
        min_n_move_mate = min(eval_of_mate.loc[i,('depth_25',slice(None))])
        min_n_move_mate_moves = correct_move_of_mate.loc[i, eval_of_mate.loc[i,]==min_n_move_mate].dropna().unique()
        min_n_move_mate_moves = intersection(min_n_move_mate_moves, move_of_mate.loc[i,('depth_25', slice(None))].tolist())
        
        #devo filtrare e salvare due casistiche sull'n_best moves:
            #1) il matto è minore di 5 mosse e tutti gli altri sono matti maggiori
            #2) il matto è maggiore di 4 mosse e l'evaluation è patta o persa
        
        white_to_move = " w " in str(all_evaluation.loc[34, 'fen'].values[0])
        if white_to_move:
            eval_of_centipawn = eval_of_centipawn.fillna(value=200)
        else: 
            eval_of_centipawn = eval_of_centipawn.fillna(value=-200)
        #tengo solo le mosse uguali alla mossa migliore se la mossa migliore è matto
        #if eval_of_mate.loc[i,('depth_25','different_levels', 20)]!=1000: #la mossa migliore è matto
        if ((eval_of_mate.loc[i,('depth_25','different_levels', 20)] <= 5) #matto in meno di 6 mosse
            or (eval_of_mate.loc[i,('depth_25','different_levels', 20)] > 5 #matto più di 5 mosse ma...
                and ((white_to_move and eval_of_centipawn.loc[i,('depth_25','n_best', 2)] < 200) #è patta o persa
                    or (not white_to_move and eval_of_centipawn.loc[i,('depth_25','n_best', 2)] > -200))) #è patta o persa
           ):
            correct_move_of_mate.loc[i,correct_move_of_mate.loc[i,:].isin(min_n_move_mate_moves) == False]=np.nan
        else:
            correct_move_of_mate.loc[i,:]=np.nan
    correct_move_of_mate.drop(columns=['fen','total_time','problem_type'], inplace=True)
    solution_map = correct_move_of_mate.notnull().astype('int')
    
    acceptable_solution = solution_map[(solution_map.iloc[:,-1]==solution_map.iloc[:,-4]) & (solution_map.iloc[:,-1]==1)]
    regret_solution = solution_map[~((solution_map.iloc[:,-1]==solution_map.iloc[:,-4]) & (solution_map.iloc[:,-1]==1))]
    
    return solution_map, acceptable_solution, regret_solution
    
def analyze_problem(acceptable_solution):
    problem_level = acceptable_solution.copy()
    idx = pd.IndexSlice
    col_name = list(problem_level.columns.levels[0].unique())
    depth_col_name = [s for s in col_name if any(xs in s for xs in ['depth'])]

    for depth_level in list(reversed(depth_col_name)):
        depth_eval = problem_level.loc[idx[:], idx[depth_level, :, :]]
        if depth_level == depth_col_name[-1]:
            problem_level['is_a_problem'] = has_one_and_only_one_solution(depth_eval, depth_level) # il livello più alto dice che ha un'unica soluzione

        if depth_level == depth_col_name[-1]:
            problem_level['is_trivial_problem'] = ~not_all_levels_are_correct(depth_eval, depth_level)
            problem_level['minimun_depth_solver'] = 100
        else:
            problem_level['is_trivial_problem'] = problem_level['is_trivial_problem'] * ~not_all_levels_are_correct(depth_eval, depth_level)

        problem_level.loc[top_level_is_correct(depth_eval, depth_level), ['minimun_depth_solver']] = int(depth_level.split("_")[1])
        #problem_level = problem_level.drop(columns=acceptable_solution.columns)
    return problem_level
    
def select_puzzle(problem_level, all_evaluation, max_move=100, min_move=2, not_start_check=False):
    is_a_problem_index = problem_level['is_a_problem'].reindex(all_evaluation.index).fillna(False)
    is_trivial_problem_index = problem_level['is_trivial_problem'].reindex(all_evaluation.index).fillna(False)

    all_evaluation_puzzles = all_evaluation[is_a_problem_index & ~is_trivial_problem_index]
    difficult = problem_level.loc[problem_level['is_a_problem'] & ~problem_level['is_trivial_problem'],['minimun_depth_solver']]
    difficult.columns = ['difficult']
    idx = pd.IndexSlice
    move = all_evaluation_puzzles.loc[idx[:], idx['depth_25','different_levels', 20, 'move', :]]
    move.columns = ['move']
    fen = all_evaluation_puzzles.loc[idx[:], idx['fen',:, :, :, :]]
    fen.columns = ['fen']
    n_move = all_evaluation_puzzles.loc[idx[:], idx['depth_25','different_levels', 20, 'mate', :]]
    n_move.columns = ['n_move']
    

    puzzle = pd.concat([fen, move, difficult, n_move], axis=1)
    #puzzle = puzzle[(puzzle['n_move']>=min_move) & (puzzle['n_move']<=max_move)]
    if not_start_check:
        puzzle = puzzle[puzzle['move'].str.find("+")==-1]
    return puzzle