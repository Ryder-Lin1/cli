#!/usr/bin/env python3
"""
Test script for the minimax chess bot
"""
import sys
import os
sys.path.append('/Users/jennylin/Documents/GitHub/cli/bin/starter/src')

import chess
from main import minimax_search, evaluate_board

def test_basic_positions():
    """Test minimax on some basic chess positions"""
    
    print("🧪 Testing Minimax Algorithm\n")
    
    # Test 1: Opening position
    print("Test 1: Opening Position (1.e4)")
    board = chess.Board()
    board.push_san('e4')
    
    print(f"Position after 1.e4:")
    print(board)
    print()
    
    best_move = minimax_search(board, depth=2)
    print(f"Minimax recommends: {board.san(best_move)}")
    print(f"Position evaluation: {evaluate_board(board)}")
    print("-" * 50)
    
    # Test 2: Tactical position (fork opportunity)
    print("Test 2: Simple Tactical Position")
    board = chess.Board()
    # Set up a position where Black can fork the king and rook
    board.set_fen("r3k3/8/8/8/8/8/4n3/R3K3 b - - 0 1")
    
    print("Position (Black can fork with Nc3+):")
    print(board)
    print()
    
    best_move = minimax_search(board, depth=3)
    print(f"Minimax recommends: {board.san(best_move)}")
    print(f"Position evaluation: {evaluate_board(board)}")
    print("-" * 50)
    
    # Test 3: Checkmate in 1
    print("Test 3: Checkmate in 1")
    board = chess.Board()
    # White can checkmate with Qh7#
    board.set_fen("r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/3P1Q2/PPP2PPP/RNB1K1NR w KQkq - 4 4")
    
    print("Position (White can mate with Qxf7#):")
    print(board)
    print()
    
    best_move = minimax_search(board, depth=2)
    print(f"Minimax recommends: {board.san(best_move)}")
    print(f"Position evaluation: {evaluate_board(board)}")
    
    # Check if it's actually checkmate
    board.push(best_move)
    if board.is_checkmate():
        print("✅ Correctly found checkmate!")
    else:
        print("❌ Did not find checkmate")
    print("-" * 50)

def test_performance():
    """Test the performance at different depths"""
    import time
    
    print("🏃 Performance Test")
    board = chess.Board()
    board.push_san('e4')
    
    for depth in [1, 2, 3]:
        start_time = time.time()
        best_move = minimax_search(board, depth=depth)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"Depth {depth}: {board.san(best_move)} in {duration:.2f}s")

if __name__ == "__main__":
    test_basic_positions()
    print()
    test_performance()
