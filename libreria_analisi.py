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
import psutil

def kill_pending_stockfish_process():
    PROCNAME = "stockfish_15_x64_avx2.exe"
    for proc in psutil.process_iter():
        # check whether the process name matches
        if proc.name() == PROCNAME:
            proc.kill()

def analyze_position(stockfishpath, board, root_moves=None, skill_level=20, hash_mb=16, time_sec=1, depth=None):
    """
    Analyze board position and return bestmove, score_cp and score_mate.
    """
    stockfishpath=r"C:\Program Files\stockfish_15_win_x64_avx2\stockfish_15_x64_avx2.exe"
    engine = chess.engine.SimpleEngine.popen_uci(stockfishpath)
    engine.configure({"Skill Level": skill_level})
    engine.configure({"Hash": hash_mb})
    if depth is None:
        result = engine.play(board, chess.engine.Limit(time=time_sec), info=chess.engine.INFO_SCORE, root_moves=root_moves)
    else:
        result = engine.play(board, chess.engine.Limit(depth=depth), info=chess.engine.INFO_SCORE, root_moves=root_moves)
    engine.quit()

    score_cp = result.info['score'].relative.score() #mate_score=32000
    score_mate = result.info['score'].relative.mate()
    bestmove = result.move

    return bestmove, score_cp, score_mate
    
def get_all_analyses_with_timeout(fen): #https://blog.finxter.com/how-to-limit-the-execution-time-of-a-function-call/
    max_wait=500
    default_value=get_all_analyses('8/8/8/8/8/8/8/8 w - - 0 1')
    default_value.loc[:] = np.nan
    default_value['fen'] = fen
    try:
        return func_timeout.func_timeout(max_wait,
                                         get_all_analyses,
                                         args=[fen])
    except func_timeout.FunctionTimedOut:
        pass
    return default_value
 
def get_all_analyses(fen):
    #if fen!='8/8/8/8/8/8/8/8 w - - 0 1':
    #    logging.info(f"analyze fen: {fen}")
    skill_levels=[10,15,20] # elo: 1750, 2160, 2570
    analysis_time_sec = 100
    analysis_depth = [5, 10, 15, 20, 25] #elo: 1900, 2231, 2563, 2894, 3000+
    #analysis_depth=[5]
    n_moves=3
    hash_mb = 128 #minore della cache size del processore (pare essere 24)

    depth_res=[]
    start_total_time = time.time()
    try:
        for d in range(len(analysis_depth)):
            #skill_levels
            t = time.time()
            score_cp_level = pd.DataFrame(np.zeros((1,len(skill_levels))), columns=skill_levels)
            score_mate_level = pd.DataFrame(np.zeros((1,len(skill_levels))), columns=skill_levels)
            move_level = pd.DataFrame(np.zeros((1,len(skill_levels))), columns=skill_levels)
            for i in range(len(skill_levels)):
                board = chess.Board(fen)
                if board.is_valid():
                    bestmove, score_cp, score_mate = analyze_position(
                        stockfishpath, board, root_moves=None, skill_level=skill_levels[i],
                        hash_mb=hash_mb, time_sec=analysis_time_sec, depth=analysis_depth[d])

                    max_s = max(skill_levels)
                    if skill_levels[i] < max_s:
                        # We will set the max skill level because we are after on the correct score of the given move.                    
                        root_move = [bestmove]
                        bestmove, score_cp, score_mate = analyze_position(
                            stockfishpath, board, root_moves=root_move, skill_level=max_s,
                            hash_mb=hash_mb, time_sec=analysis_time_sec, depth=analysis_depth[d])

                    score_cp_level[skill_levels[i]]=score_cp
                    score_mate_level[skill_levels[i]]=score_mate
                    move_level[skill_levels[i]]=board.san(bestmove)                
                else:
                    score_cp_level = score_cp_level.replace([0, 0.0], np.nan)
                    score_mate_level = score_mate_level.replace([0, 0.0], np.nan)
                    move_level = move_level.replace([0, 0.0], np.nan)
            elapsed_time_levels = time.time() - t

            #n_best
            t = time.time()
            score_cp_n_best = pd.DataFrame(np.zeros((1,n_moves)), columns=range(n_moves,0,-1))
            score_mate_n_best = pd.DataFrame(np.zeros((1,n_moves)), columns=range(n_moves,0,-1))
            move_n_best = pd.DataFrame(np.zeros((1,n_moves)), columns=range(n_moves,0,-1))
            board = chess.Board(fen)
            if board.is_valid():
                engine = chess.engine.SimpleEngine.popen_uci(stockfishpath)
                res_n_best = engine.analyse(board, chess.engine.Limit(depth=analysis_depth[d]), multipv=n_moves)#, info=chess.engine.INFO_SCORE)
                engine.quit()
                for i in range(n_moves):
                    res_splitted=str(res_n_best[i]['score']).split("(")
                    value_res=int(str(res_splitted[2].split(")")[0]))
                    if res_splitted[1] == 'Cp':
                        score_cp_n_best[i+1] = value_res
                        score_mate_n_best[i+1] = None
                    elif res_splitted[1] == 'Mate':
                        score_cp_n_best[i+1] = None
                        score_mate_n_best[i+1] = value_res
                    move_n_best[i+1] = board.san(res_n_best[i]['pv'][0])
            else:
                score_cp_n_best = score_cp_n_best.replace([0, 0.0], np.nan)
                score_mate_n_best = score_mate_n_best.replace([0, 0.0], np.nan)
                move_n_best = move_n_best.replace([0, 0.0], np.nan)
            elapsed_time_n_best = time.time() - t

            levels_res = pd.concat([score_cp_level, score_mate_level, move_level], axis = 1, keys=['centipawn','mate', 'move']) 
            levels_res['time'] = elapsed_time_levels
            n_best_res = pd.concat([score_cp_n_best, score_mate_n_best, move_n_best], axis = 1, keys=['centipawn','mate', 'move'])
            n_best_res['time'] = elapsed_time_n_best
            this_dept_res = pd.concat([levels_res, n_best_res], axis = 1, keys=['different_levels','n_best'])
            depth_res.append(this_dept_res)

        depth_res=pd.concat(depth_res, axis = 1, keys=[f'depth_{d}' for d in analysis_depth])
        depth_res.columns.names= ['depth_level','method','score_move','grade']
        depth_res.index.names= ['board']
        depth_res['fen'] = fen
        depth_res['total_time'] = time.time() - start_total_time
        kill_pending_stockfish_process()
        return depth_res.reorder_levels(['depth_level','method','grade','score_move'], axis=1)
    except Exception as ex:
        print("An exception occurred")
        print(ex)
        print(f"fen: {fen}")    
        kill_pending_stockfish_process()        
        return get_all_analyses('8/8/8/8/8/8/8/8 w - - 0 1')
        
from ipywidgets import IntProgress
from IPython.display import display

def analyze_all_fens(fens, problem_type=666, multi_thread=True):
    columns_df = ['first_cp','second_cp','third_cp',
                     'first_mate_in', 'second_mate_in', 'third_mate_in',
                     'lev_first_cp','lev_second_cp','lev_third_cp',
                     'lev_first_mate_in', 'lev_second_mate_in', 'lev_third_mate_in',
                     'fen', 'time']
    tt2 = time.time()
    if multi_thread:
        num_cores = multiprocessing.cpu_count()
        #num_real_cores = num_cores/2
        all_evaluation = Parallel(n_jobs=num_cores)(delayed(get_all_analyses_with_timeout)(fens[f]) for f in range(len(fens)))
        all_evaluation=pd.DataFrame(np.concatenate(all_evaluation), columns = all_evaluation[0].columns)
    else:
        all_evaluation=[]
        max_count = len(fens)
        k = IntProgress(min=0, max=max_count)
        display(k) 
        for f in range(len(fens)):
            all_evaluation.append(get_all_analyses_with_timeout(fens[f]))
            k.value += 1
        all_evaluation=pd.DataFrame(np.concatenate(all_evaluation), columns = all_evaluation[0].columns)
    total_time = time.time() - tt2 
    all_evaluation['problem_type']=problem_type
    return all_evaluation, total_time
    
def parallel_analyze_in_chunks(df, _problem_type, file_name, folder_output, multiplier_size_chunk = 1, multi_thread_=True,
                               restart_from_chunk_n = 0, reverse_order_chunk=False):
        
    try:
        file_path_output = os.path.join(folder_output, file_name)
        temp_dir = os.path.join(folder_output, 'temp')
        num_cores = multiprocessing.cpu_count()
        n_parts = math.ceil((len(df)/(num_cores*multiplier_size_chunk)))
        df_splitted=np.array_split(df['fen'], n_parts)
        if reverse_order_chunk:
            rangeee = range(n_parts-1,restart_from_chunk_n,-1)
        else:
            rangeee = range(restart_from_chunk_n,n_parts)
        for i in rangeee:
            all_evaluation_df_splitted, total_time = analyze_all_fens(df_splitted[i].reset_index(drop=True), problem_type=_problem_type, multi_thread=multi_thread_)
            print(f'{file_name}, part {i} done. Time={total_time}')
            part_file_path = os.path.join(temp_dir, f'{file_name}_temp_{i}.pk1')
            all_evaluation_df_splitted.to_pickle(part_file_path)

        all_evaluation_df_splitted=[]
        for i in range(0,n_parts):
            part_file_path = os.path.join(temp_dir, f'{file_name}_temp_{i}.pk1')
            with open(part_file_path, 'rb') as pickle_file:
                all_evaluation_df_splitted.append(pickle.load(pickle_file))
            pickle_file.close()
            os.remove(part_file_path)
            all_evaluation = pd.concat(all_evaluation_df_splitted, ignore_index=True) 

        all_evaluation.to_pickle(file_path_output)
    except Exception as e: 
        print(e)

