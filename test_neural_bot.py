#!/usr/bin/env python3
"""
Test the chess bot with neural network integration
"""
import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'bin', 'starter', 'src'))

try:
    from utils import GameContext
    from main import test_func
    import chess
    
    print("🎮 Testing Chess Bot with Neural Network Integration")
    print("=" * 60)
    
    # Create a test board with an interesting position
    board = chess.Board("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 4")
    
    print(f"🏁 Test position: {board.fen}")
    print("📋 Board:")
    print(board)
    print()
    
    # Create game context
    ctx = GameContext()
    ctx.board = board
    
    # Test the bot's move selection
    print("🤖 Bot is thinking...")
    try:
        move = test_func(ctx)
        print(f"🎯 Bot selected move: {board.san(move)}")
        print(f"✅ Neural network integration test completed!")
        
        # Show the position after the move
        board.push(move)
        print("\n📋 Position after move:")
        print(board)
        
    except Exception as e:
        print(f"❌ Error during move selection: {e}")
        import traceback
        traceback.print_exc()

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the correct directory and dependencies are installed")
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
