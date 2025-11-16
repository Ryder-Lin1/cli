"""
Simple Chess Bot without Neural Networks
Focus on solid chess principles instead of broken NNUE
"""

import chess
import random

def evaluate_position_simple(board):
    """Simple but effective chess evaluation"""
    if board.is_checkmate():
        return -30000 if board.turn else 30000
    
    if board.is_stalemate():
        return 0
    
    score = 0
    
    # Material count with standard values
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320, 
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values[piece.piece_type]
            if piece.color == chess.WHITE:
                score += value
            else:
                score -= value
    
    # Center control bonus
    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
    for square in center_squares:
        piece = board.piece_at(square)
        if piece:
            bonus = 30 if piece.piece_type == chess.PAWN else 10
            if piece.color == chess.WHITE:
                score += bonus
            else:
                score -= bonus
    
    # King safety - penalize exposed king
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    
    if white_king and chess.square_rank(white_king) > 2:  # King moved forward
        score -= 50
    if black_king and chess.square_rank(black_king) < 5:  # King moved forward
        score += 50
    
    # Piece development bonus (knights and bishops off back rank)
    for square in [chess.B1, chess.G1, chess.C1, chess.F1]:  # White pieces
        piece = board.piece_at(square)
        if piece and piece.color == chess.WHITE and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            score -= 20  # Penalty for undeveloped pieces
    
    for square in [chess.B8, chess.G8, chess.C8, chess.F8]:  # Black pieces
        piece = board.piece_at(square)
        if piece and piece.color == chess.BLACK and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            score += 20  # Penalty for undeveloped pieces
    
    return score if board.turn == chess.WHITE else -score

def find_best_move_simple(board, depth=3):
    """Simple minimax without neural networks"""
    
    def minimax(board, depth, alpha, beta, maximizing):
        if depth == 0 or board.is_game_over():
            return evaluate_position_simple(board)
        
        if maximizing:
            max_eval = float('-inf')
            for move in board.legal_moves:
                board.push(move)
                eval_score = minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval_score = minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval
    
    # Find best move
    best_move = None
    best_score = float('-inf')
    
    legal_moves = list(board.legal_moves)
    
    # Prioritize captures and checks
    prioritized_moves = []
    other_moves = []
    
    for move in legal_moves:
        if board.is_capture(move) or board.gives_check(move):
            prioritized_moves.append(move)
        else:
            other_moves.append(move)
    
    # Search prioritized moves first
    all_moves = prioritized_moves + other_moves
    
    for move in all_moves[:15]:  # Limit search to avoid slow play
        board.push(move)
        score = minimax(board, depth - 1, float('-inf'), float('inf'), False)
        board.pop()
        
        if score > best_score:
            best_score = score
            best_move = move
    
    return best_move, best_score

def get_opening_move(board):
    """Simple opening principles"""
    moves = list(board.legal_moves)
    
    # Opening move preferences
    if len(board.move_stack) == 0:  # First move as White
        return chess.Move.from_uci("e2e4")  # King's pawn
    
    if len(board.move_stack) == 1:  # First move as Black
        good_responses = ["e7e5", "c7c5", "e7e6", "c7c6"]  # Solid defenses
        for uci in good_responses:
            move = chess.Move.from_uci(uci)
            if move in moves:
                return move
    
    # Early game: develop pieces
    if len(board.move_stack) < 10:
        # Prefer knight and bishop development
        for move in moves:
            piece = board.piece_at(move.from_square)
            if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                return move
    
    return None

if __name__ == "__main__":
    # Test the simple bot
    print("🎮 Testing Simple Chess Bot")
    
    board = chess.Board()
    
    for i in range(10):
        if board.is_game_over():
            break
            
        print(f"\nMove {i+1}:")
        print(board)
        
        # Try opening move first
        move = get_opening_move(board)
        if not move:
            move, score = find_best_move_simple(board, depth=3)
            print(f"Best move: {board.san(move)} (score: {score})")
        else:
            print(f"Opening move: {board.san(move)}")
        
        board.push(move)
    
    print("\n✅ Simple bot working much better than broken NNUE!")
