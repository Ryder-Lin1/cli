#!/usr/bin/env python3
"""Test the bot against all games in the database"""

import chess
import chess.pgn
import sys
import os
from collections import defaultdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from main import (
    find_best_move_simple,
    query_opening_database,
    black_openings,
    evaluate_position_simple
)

def test_bot_on_database():
    """Run bot through all database games and analyze performance"""
    
    print(f"\n{'='*60}")
    print(f"🧪 Testing bot on {len(black_openings)} database games")
    print(f"{'='*60}\n")
    
    stats = {
        'total_games': 0,
        'total_positions': 0,
        'database_matches': 0,
        'engine_moves': 0,
        'correct_moves': 0,
        'move_depth': defaultdict(int),
        'errors': 0
    }
    
    # Test on first 100 games (change this to len(black_openings) for all games)
    games_to_test = min(100, len(black_openings))
    
    for game_idx, game in enumerate(black_openings[:games_to_test]):
        try:
            board = chess.Board()
            game_moves = list(game.mainline_moves())
            
            stats['total_games'] += 1
            
            # Test up to move 20 (or end of game)
            max_moves = min(20, len(game_moves))
            
            for move_num, correct_move in enumerate(game_moves[:max_moves]):
                stats['total_positions'] += 1
                stats['move_depth'][move_num] += 1
                
                # Try database first
                db_move = query_opening_database(board)
                
                if db_move and db_move == correct_move:
                    stats['database_matches'] += 1
                    stats['correct_moves'] += 1
                    board.push(correct_move)
                    continue
                
                # Try engine
                try:
                    engine_move = find_best_move_simple(board, depth=3)
                    stats['engine_moves'] += 1
                    
                    if engine_move == correct_move:
                        stats['correct_moves'] += 1
                    
                except Exception as e:
                    stats['errors'] += 1
                
                # Play the correct move to continue
                board.push(correct_move)
            
            # Progress indicator
            if (game_idx + 1) % 10 == 0:
                accuracy = (stats['correct_moves'] / stats['total_positions'] * 100) if stats['total_positions'] > 0 else 0
                print(f"📊 Progress: {game_idx + 1}/{games_to_test} games | Accuracy: {accuracy:.1f}%")
        
        except Exception as e:
            print(f"❌ Error on game {game_idx}: {e}")
            stats['errors'] += 1
            continue
    
    # Final statistics
    print(f"\n{'='*60}")
    print(f"📈 FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Games tested: {stats['total_games']}")
    print(f"Positions analyzed: {stats['total_positions']}")
    print(f"Database matches: {stats['database_matches']} ({stats['database_matches']/stats['total_positions']*100:.1f}%)")
    print(f"Engine moves: {stats['engine_moves']} ({stats['engine_moves']/stats['total_positions']*100:.1f}%)")
    print(f"Correct moves: {stats['correct_moves']} ({stats['correct_moves']/stats['total_positions']*100:.1f}%)")
    print(f"Errors: {stats['errors']}")
    
    print(f"\n📊 Accuracy by move number:")
    for move_num in sorted(stats['move_depth'].keys())[:20]:
        count = stats['move_depth'][move_num]
        print(f"  Move {move_num + 1}: {count} positions")
    
    print(f"\n{'='*60}\n")

def quick_test():
    """Quick test of a single position"""
    print("\n🧪 Quick test of bot...")
    
    board = chess.Board()
    print(f"Starting position: {board.fen()}")
    
    # Test first move
    db_move = query_opening_database(board)
    if db_move:
        print(f"✅ Database move: {board.san(db_move)}")
    else:
        print(f"❌ No database move found")
    
    engine_move = find_best_move_simple(board, depth=3)
    if engine_move:
        print(f"✅ Engine move: {board.san(engine_move)}")
    else:
        print(f"❌ No engine move found")
    
    print()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test chess bot on database')
    parser.add_argument('--quick', action='store_true', help='Quick test only')
    parser.add_argument('--full', action='store_true', help='Test all games (slow)')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_test()
    else:
        if args.full:
            print("⚠️  Warning: Testing all games will take a long time!")
            response = input("Continue? (y/n): ")
            if response.lower() != 'y':
                sys.exit(0)
        
        test_bot_on_database()
