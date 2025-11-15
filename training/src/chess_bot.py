"""
Chess Bot with NNUE Evaluation
Simple chess bot that uses NNUE for position evaluation
"""

import chess
import random
import time
from nnue import evaluate_position


class NNUEChessBot:
    """
    Chess bot powered by NNUE neural network evaluation
    """
    
    def __init__(self, model_path="../models/nnue_model.pt", search_depth=3):
        self.model_path = model_path
        self.search_depth = search_depth
        
        # Statistics
        self.positions_evaluated = 0
        self.search_time = 0.0
        
        print(f"🤖 NNUE Chess Bot initialized")
        print(f"📁 Model path: {model_path}")
        print(f"🔍 Search depth: {search_depth}")
    
    def evaluate_position(self, board):
        """Evaluate position using NNUE"""
        self.positions_evaluated += 1
        return evaluate_position(board, self.model_path)
    
    def minimax(self, board, depth, alpha, beta, maximizing_player):
        """
        Minimax search with alpha-beta pruning and NNUE evaluation
        """
        # Terminal cases
        if depth == 0 or board.is_game_over():
            if board.is_checkmate():
                return -20000 if maximizing_player else 20000
            elif board.is_stalemate() or board.is_insufficient_material():
                return 0
            else:
                evaluation = self.evaluate_position(board)
                return evaluation if maximizing_player else -evaluation
        
        if maximizing_player:
            max_eval = float('-inf')
            for move in board.legal_moves:
                board.push(move)
                eval_score = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                
                if beta <= alpha:
                    break  # Alpha-beta pruning
            
            return max_eval
        else:
            min_eval = float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval_score = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                
                if beta <= alpha:
                    break  # Alpha-beta pruning
            
            return min_eval
    
    def get_best_move(self, board):
        """
        Find the best move using NNUE-powered minimax search
        """
        start_time = time.time()
        self.positions_evaluated = 0
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        print(f"🔍 Searching {len(legal_moves)} moves at depth {self.search_depth}...")
        
        best_move = None
        best_score = float('-inf')
        
        # Evaluate each move
        for i, move in enumerate(legal_moves):
            board.push(move)
            
            # Search with minimax
            score = self.minimax(
                board, 
                self.search_depth - 1, 
                float('-inf'), 
                float('inf'), 
                False  # Opponent's turn after our move
            )
            
            board.pop()
            
            if score > best_score:
                best_score = score
                best_move = move
            
            # Progress indicator
            if len(legal_moves) > 10 and i % 3 == 0:
                print(f"  📊 {i+1}/{len(legal_moves)} moves evaluated...")
        
        # Calculate search statistics
        search_time = time.time() - start_time
        self.search_time += search_time
        
        print(f"🎯 Best move: {board.san(best_move)}")
        print(f"📈 Evaluation: {best_score:.2f} centipawns")
        print(f"⏱️ Search time: {search_time:.2f}s")
        print(f"🔢 Positions evaluated: {self.positions_evaluated}")
        
        return best_move
    
    def play_move(self, board):
        """
        Play a move on the given board
        Returns the move played
        """
        print(f"\n🤖 {'White' if board.turn else 'Black'} to move")
        print(f"📝 Position: {board.fen()}")
        
        # Quick tactical checks
        move = self._find_tactical_move(board)
        if move:
            print(f"⚡ Tactical move found: {board.san(move)}")
            return move
        
        # NNUE search
        move = self.get_best_move(board)
        return move
    
    def _find_tactical_move(self, board):
        """Look for immediate tactical opportunities"""
        legal_moves = list(board.legal_moves)
        
        # Check for checkmate in 1
        for move in legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move
            board.pop()
        
        # Check for captures
        captures = [move for move in legal_moves if board.is_capture(move)]
        if captures:
            # Evaluate captures quickly
            best_capture = None
            best_value = float('-inf')
            
            for move in captures:
                captured_piece = board.piece_at(move.to_square)
                if captured_piece:
                    # Simple material values
                    piece_values = {
                        chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
                        chess.ROOK: 500, chess.QUEEN: 900
                    }
                    
                    value = piece_values.get(captured_piece.piece_type, 0)
                    if value > best_value:
                        best_value = value
                        best_capture = move
            
            # Only return capture if it's valuable
            if best_value >= 300:  # Knight or better
                return best_capture
        
        return None
    
    def get_statistics(self):
        """Get bot performance statistics"""
        return {
            'positions_evaluated': self.positions_evaluated,
            'total_search_time': self.search_time,
            'avg_positions_per_second': self.positions_evaluated / max(self.search_time, 0.01)
        }


def play_game():
    """Play a game with the NNUE bot"""
    print("🎮 Starting NNUE Chess Game")
    print("=" * 40)
    
    # Create bot
    bot = NNUEChessBot()
    
    # Create board
    board = chess.Board()
    
    # Game loop
    move_count = 0
    while not board.is_game_over() and move_count < 50:  # Limit moves for demo
        print(f"\n--- Move {move_count + 1} ---")
        print(board)
        
        # Bot plays
        move = bot.play_move(board)
        if move:
            board.push(move)
            move_count += 1
        else:
            print("❌ No move available")
            break
        
        # Check game state
        if board.is_checkmate():
            winner = "White" if board.turn == chess.BLACK else "Black"
            print(f"🏆 Checkmate! {winner} wins!")
            break
        elif board.is_stalemate():
            print("🤝 Stalemate!")
            break
        elif board.is_insufficient_material():
            print("🤝 Draw by insufficient material!")
            break
    
    # Final statistics
    print("\n📊 Game Statistics:")
    stats = bot.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")
    
    print(f"\n🏁 Final position:")
    print(board)


def test_nnue_bot():
    """Test the NNUE bot on a few positions"""
    print("🧪 Testing NNUE Chess Bot")
    print("=" * 30)
    
    bot = NNUEChessBot(search_depth=2)  # Shallow search for testing
    
    # Test positions
    test_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",  # After e4
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 2 2"  # After e4 e5 Nc6
    ]
    
    for i, fen in enumerate(test_fens):
        print(f"\n🎯 Test Position {i + 1}:")
        board = chess.Board(fen)
        print(board)
        
        move = bot.get_best_move(board)
        if move:
            print(f"✅ Recommended move: {board.san(move)}")
        else:
            print("❌ No move found")
    
    print("\n✅ NNUE bot testing complete!")


if __name__ == "__main__":
    # Run tests first
    test_nnue_bot()
    
    # Then play a short game
    print("\n" + "=" * 50)
    play_game()
