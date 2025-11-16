try:
    from .utils import chess_manager, GameContext
except ImportError:
    from utils import chess_manager, GameContext
from chess import Move
import chess
import random
import time
import chess.pgn
import os
import sys
import torch
import math

# Import NNUE module from same directory
try:
    from .nnue_local import NNUEModel, NNUEFeatures
except ImportError:
    from nnue_local import NNUEModel, NNUEFeatures

# Piece values for tactical evaluation
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}

# NNUE-based evaluation (REQUIRED for competition)
def evaluate_position_nnue(board):
    """
    Evaluate position using trained NNUE neural network
    This is the PRIMARY evaluation method - neural network is CRITICAL
    """
    # Terminal position checks
    if board.is_checkmate():
        return -30000 if board.turn else 30000
    
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    
    try:
        # Extract NNUE features from board
        white_features, black_features, buckets = NNUEFeatures.board_to_nnue_features(board)
        print(f"[DEBUG] White features: {white_features}")
        print(f"[DEBUG] Black features: {black_features}")
        print(f"[DEBUG] Buckets: {buckets}")
        if white_features is None:
            # Fallback only if feature extraction fails
            return 0
        # Run neural network inference
        with torch.no_grad():
            stm = torch.tensor([1.0 if board.turn == chess.WHITE else -1.0], device=device)
            evaluation = nnue_model(
                [white_features], 
                [black_features], 
                [buckets[0]], 
                [buckets[1]], 
                stm
            )
            print(f"[DEBUG] Raw NNUE output: {evaluation}")
            # Convert to centipawns
            score = evaluation.item() * 100
            print(f"[DEBUG] Centipawn score: {score}")
            # Return from current player's perspective
            return score if board.turn == chess.WHITE else -score
            
    except Exception as e:
        print(f"⚠️ NNUE evaluation error: {e}")
        # If neural network fails, bot cannot function (as per competition rules)
        return 0

def negamax(board, depth, alpha, beta):
    """Negamax search with alpha-beta pruning using NNUE evaluation"""
    if depth == 0 or board.is_game_over():
        return evaluate_position_nnue(board)
    
    max_score = float('-inf')
    for move in board.legal_moves:
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha)
        board.pop()
        
        max_score = max(max_score, score)
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    
    return max_score

# MCTS Node class for tree search
class MCTSNode:
    def __init__(self, board, parent=None, move=None):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_moves = list(board.legal_moves)
        
    def is_fully_expanded(self):
        return len(self.untried_moves) == 0
    
    def is_terminal(self):
        return self.board.is_game_over()
    
    def best_child(self, c_param=1.4):
        """Select best child using UCB1 formula"""
        choices_weights = []
        for child in self.children:
            if child.visits == 0:
                weight = float('inf')
            else:
                # UCB1 formula: exploitation + exploration
                exploitation = child.value / child.visits
                exploration = c_param * math.sqrt(math.log(self.visits) / child.visits)
                weight = exploitation + exploration
            choices_weights.append(weight)
        
        return self.children[choices_weights.index(max(choices_weights))]
    
    def expand(self):
        """Expand a random untried move"""
        move = self.untried_moves.pop()
        new_board = self.board.copy()
        new_board.push(move)
        child_node = MCTSNode(new_board, parent=self, move=move)
        self.children.append(child_node)
        return child_node
    
    def backpropagate(self, result):
        """Backpropagate the result up the tree"""
        self.visits += 1
        self.value += result
        if self.parent:
            # Flip result for parent (opponent's perspective)
            self.parent.backpropagate(-result)

def mcts_search(board, num_simulations=100, time_limit=2.0):
    """
    Monte Carlo Tree Search guided by NNUE neural network
    Uses NNUE for position evaluation instead of random playouts
    """
    start_time = time.time()
    root = MCTSNode(board)
    
    simulations_done = 0
    
    for i in range(num_simulations):
        # Check time limit
        if time.time() - start_time > time_limit:
            print(f"⏱️ MCTS stopped after {simulations_done} simulations (time limit)")
            break
        
        # Selection: traverse tree using UCB1
        node = root
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child()
        
        # Expansion: add a new child node
        if not node.is_terminal() and not node.is_fully_expanded():
            node = node.expand()
        
        # Simulation: use NNUE to evaluate position (no random playout needed!)
        if node.is_terminal():
            # Terminal node evaluation
            if node.board.is_checkmate():
                result = -1.0  # Loss for player who just moved
            else:
                result = 0.0  # Draw
        else:
            # Use NNUE neural network to evaluate position
            nnue_eval = evaluate_position_nnue(node.board)
            # Normalize to [-1, 1] range
            result = math.tanh(nnue_eval / 200.0)
        
        # Backpropagation: update statistics up the tree
        node.backpropagate(result)
        simulations_done += 1
    
    # Select best move based on visit counts (most robust)
    if not root.children:
        return None
    
    best_child = max(root.children, key=lambda c: c.visits)
    
    # Log MCTS statistics
    print(f"🌲 MCTS completed {simulations_done} simulations in {time.time() - start_time:.2f}s")
    print(f"   Best move: {board.san(best_child.move)} (visits: {best_child.visits}, value: {best_child.value/best_child.visits:.3f})")
    
    return best_child.move

# Track move history to avoid repetition
move_history = []
position_history = {}

def find_best_move_simple(board, depth=3):
    """Minimax with alpha-beta pruning and repetition avoidance"""
    global move_history, position_history
    
    best_move = None
    alpha = float('-inf')
    beta = float('inf')
    
    # Get all legal moves
    legal_moves = list(board.legal_moves)
    
    # Track current position
    current_fen = board.fen().split(' ')[0]  # Just piece positions
    
    # Prioritize captures, but only if they're good trades
    prioritized_moves = []
    development_moves = []
    other_moves = []
    repetitive_moves = []
    
    for move in legal_moves:
        # Check if this move leads to repetition
        board.push(move)
        future_fen = board.fen().split(' ')[0]
        board.pop()
        
        is_repetitive = position_history.get(future_fen, 0) >= 1
        
        captured_piece = board.piece_at(move.to_square)
        moving_piece = board.piece_at(move.from_square)
        
        # Identify development moves
        is_development = False
        if moving_piece and moving_piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            from_rank = chess.square_rank(move.from_square)
            # Knights/bishops moving from back rank
            if (moving_piece.color == chess.WHITE and from_rank == 0) or (moving_piece.color == chess.BLACK and from_rank == 7):
                is_development = True
        
        if is_repetitive:
            repetitive_moves.append(move)
        elif captured_piece and moving_piece:
            # Only prioritize if we're capturing something of equal or greater value
            if PIECE_VALUES[captured_piece.piece_type] >= PIECE_VALUES[moving_piece.piece_type]:
                prioritized_moves.append(move)
            else:
                other_moves.append(move)
        elif is_development:
            development_moves.append(move)
        else:
            other_moves.append(move)
    
    # Search order: good captures -> development -> other moves -> repetitive moves (last resort)
    ordered_moves = (prioritized_moves[:15] + development_moves[:15] + 
                     other_moves[:20] + repetitive_moves)
    
    for move in ordered_moves:
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha)
        board.pop()
        
        if score > alpha:
            alpha = score
            best_move = move
    
    # Update position history
    if best_move:
        board.push(best_move)
        final_fen = board.fen().split(' ')[0]
        position_history[final_fen] = position_history.get(final_fen, 0) + 1
        board.pop()
        
        # Keep history manageable
        if len(position_history) > 50:
            # Remove oldest entries
            oldest_keys = list(position_history.keys())[:10]
            for key in oldest_keys:
                del position_history[key]
    
    return best_move

def get_opening_move(board):
    """NO DATABASE MODE - Returns None to force engine calculation"""
    # Competition rules forbid opening databases
    # All moves must be calculated from scratch
    return None

def query_opening_database(board):
    """NO DATABASE - Competition forbids databases"""
    # Competition rules: no databases allowed
    # All moves must use neural network evaluation
    return None

def get_piece_value(piece):
    """Get the value of a chess piece"""
    if piece is None:
        return 0
    return PIECE_VALUES.get(piece.piece_type, 0)

def calculate_exchange_value(board, move):
    """Calculate the material value of a capture exchange"""
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0  # King can't really be captured
    }
    
    captured_piece = board.piece_at(move.to_square)
    capturing_piece = board.piece_at(move.from_square)
    
    if not captured_piece or not capturing_piece:
        return 0
    
    # Basic exchange: value of captured piece minus value of our piece if recaptured
    captured_value = piece_values.get(captured_piece.piece_type, 0)
    our_piece_value = piece_values.get(capturing_piece.piece_type, 0)
    
    # Simple calculation: assume we might lose our piece in return
    return captured_value - our_piece_value

def is_piece_protected(board, square):
    """Check if a piece on the given square is protected by opponent"""
    piece = board.piece_at(square)
    if not piece:
        return False
    
    # Check if any opponent piece can capture this square
    opponent_color = not piece.color
    for opponent_square in chess.SQUARES:
        opponent_piece = board.piece_at(opponent_square)
        if opponent_piece and opponent_piece.color == opponent_color:
            # Check if this opponent piece can move to our target square
            test_move = chess.Move(opponent_square, square)
            if test_move in board.legal_moves:
                return True
    return False

def find_best_capture(board):
    """Find the best capture move based on material exchange, avoiding protected pieces"""
    legal_moves = list(board.legal_moves)
    capture_moves = [move for move in legal_moves if board.is_capture(move)]
    
    if not capture_moves:
        return None
    
    best_move = None
    best_score = float('-inf')
    
    for move in capture_moves:
        target_square = move.to_square
        
        # Skip if the target piece is protected
        if is_piece_protected(board, target_square):
            print(f"⚠️ Skipping capture on {chess.square_name(target_square)} - piece is protected")
            continue
        
        # Calculate the exchange value
        exchange_value = calculate_exchange_value(board, move)
        
        # Only consider captures that are beneficial or at least equal
        if exchange_value >= 0:
            if exchange_value > best_score:
                best_score = exchange_value
                best_move = move
                print(f"✅ Good unprotected capture: {move} (value: +{exchange_value})")
    
    return best_move

def find_checks_and_threats(board):
    """Look for checks and immediate threats"""
    legal_moves = list(board.legal_moves)
    
    # Prioritize checks
    checks = [move for move in legal_moves if board.gives_check(move)]
    if checks:
        check_move = random.choice(checks)
        print(f"Found check: {check_move}")
        return check_move
    
    return None

# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etc.

# NO DATABASE LOADING - Competition rules forbid databases
# All evaluation must use trained neural network

# Load trained NNUE model (REQUIRED for competition)
print("🧠 Loading trained NNUE model...")
nnue_model = NNUEModel()
device = torch.device('cpu')  # Use CPU for fast inference
nnue_model.to(device)

# Load trained weights
weights_path = os.path.join(os.path.dirname(__file__), "weights", "nnue_model.pt")
if os.path.exists(weights_path):
    nnue_model.load_state_dict(torch.load(weights_path, map_location=device))
    nnue_model.eval()
    print(f"✅ NNUE model loaded from {weights_path}")
else:
    print(f"⚠️ Warning: NNUE weights not found at {weights_path}")
    print("   Please train the model first using train_nnue_local.py")

print("♟️ Chess bot initialized - NNUE Neural Network Mode")
print("   Using NNUE evaluation with minimax search")



def evaluate_board(board):
    """Evaluate the current board position"""
    if board.is_checkmate():
        return -20000 if board.turn else 20000  # Negative if current player is checkmated
    
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    
    score = 0
    
    # Material evaluation
    for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        white_pieces = len(board.pieces(piece_type, chess.WHITE))
        black_pieces = len(board.pieces(piece_type, chess.BLACK))
        score += (white_pieces - black_pieces) * PIECE_VALUES[piece_type]
    
    # Positional bonuses
    # Center control
    center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
    for square in center_squares:
        piece = board.piece_at(square)
        if piece:
            if piece.color == chess.WHITE:
                score += 10
            else:
                score -= 10
    
    # Mobility (number of legal moves)
    legal_moves = len(list(board.legal_moves))
    if board.turn == chess.WHITE:
        score += legal_moves * 2
    else:
        score -= legal_moves * 2
    
    # King safety
    if board.is_check():
        if board.turn == chess.WHITE:
            score -= 50
        else:
            score += 50
    
    # Return score from current player's perspective
    return score if board.turn == chess.WHITE else -score

def minimax(board, depth, alpha, beta, maximizing_player):
    """Minimax algorithm with alpha-beta pruning"""
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)
    
    if maximizing_player:
        max_eval = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, alpha, beta, False)
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
            eval_score = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break  # Alpha-beta pruning
        return min_eval

def minimax_search(board, depth=3):
    """Find the best move using minimax algorithm"""
    best_move = None
    best_score = float('-inf')
    
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    
    print(f"🧠 Analyzing {len(legal_moves)} moves at depth {depth}...")
    
    for i, move in enumerate(legal_moves):
        board.push(move)
        # We're maximizing for the current player
        score = minimax(board, depth - 1, float('-inf'), float('inf'), False)
        board.pop()
        
        if score > best_score:
            best_score = score
            best_move = move
        
        # Progress indicator for longer searches
        if i % 5 == 0 and len(legal_moves) > 10:
            print(f"  📊 Evaluated {i+1}/{len(legal_moves)} moves...")
    
    print(f"🎯 Best move: {board.san(best_move)} (score: {best_score})")
    return best_move

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    """
    Chess bot move selection - MCTS + NNUE NEURAL NETWORK MODE
    Neural network is CRITICAL - guides MCTS tree search and evaluates all positions
    """
    try:
        # Wait 2 seconds after opponent's move to analyze position
        print("⏳ Analyzing opponent's move for 2 seconds...")
        time.sleep(2)
        
        board = ctx.board.copy()
        
        print(f"🧠 NNUE-guided MCTS for position: {board.fen()}")
        
        # Check game phase to determine search strategy
        move_number = board.fullmove_number
        num_pieces = len(board.piece_map())
        
        # Use MCTS for middle game and complex positions (most effective)
        if 10 <= num_pieces <= 25 and move_number > 5:
            print("🌲 Using MCTS with NNUE evaluation...")
            try:
                # Adjust simulations based on position complexity
                num_sims = 150 if num_pieces > 15 else 100
                best_move = mcts_search(board, num_simulations=num_sims, time_limit=2.0)
                
                if best_move:
                    board.push(best_move)
                    eval_score = evaluate_position_nnue(board)
                    board.pop()
                    
                    print(f"🎯 MCTS+NNUE move: {ctx.board.san(best_move)} (eval: {eval_score:.0f})")
                    ctx.logProbabilities({best_move: 1.0})
                    return best_move
            except Exception as e:
                print(f"⚠️ MCTS failed, falling back to minimax: {e}")
        
        # Use minimax for opening, endgame, or as fallback
        print("♟️ Using NNUE-guided minimax...")
        try:
            # Quick search depth (takes about 2 seconds)
            best_move = find_best_move_simple(board, depth=2)
            if best_move:
                # Get NNUE evaluation of the resulting position
                board.push(best_move)
                eval_score = evaluate_position_nnue(board)
                board.pop()
                
                print(f"🎯 NNUE minimax move: {ctx.board.san(best_move)} (eval: {eval_score:.0f})")
                ctx.logProbabilities({best_move: 1.0})
                return best_move
        except Exception as e:
            print(f"⚠️ Minimax failed: {e}")
        
        # Fallback: evaluate all legal moves with NNUE
        print("🔄 Using direct NNUE evaluation fallback...")
        legal_moves = list(board.legal_moves)
        if legal_moves:
            best_move = None
            best_eval = float('-inf')
            
            for move in legal_moves:
                board.push(move)
                eval_score = evaluate_position_nnue(board)
                board.pop()
                
                if eval_score > best_eval:
                    best_eval = eval_score
                    best_move = move
            
            if best_move:
                print(f"� NNUE fallback move: {ctx.board.san(best_move)} (eval: {best_eval:.0f})")
                ctx.logProbabilities({best_move: 1.0})
                return best_move
        
        print("❌ No legal moves found!")
        return None
        
    except Exception as e:
        print(f"❌ Critical error: {e}")
        print("❌ Cannot function without neural network")
        return None


def analyze_move_quality(ctx: GameContext, move):
    """
    Analyze the quality of a move using local evaluation
    """
    try:
        # Get evaluation before the move
        eval_before = evaluate_board(ctx.board)
        
        # Make the move temporarily to evaluate the resulting position
        ctx.board.push(move)
        
        # Get evaluation after the move  
        eval_after = evaluate_board(ctx.board)
        
        # Calculate move quality
        last_move = ctx.board.peek()  # Get the last move played
        eval_change = eval_after - eval_before
        
        # Interpret the evaluation change
        if abs(eval_change) < 50:
            quality = "⚪ Neutral"
        elif eval_change > 100:
            quality = "🟢 Good"  
        elif eval_change > 200:
            quality = "🔥 Excellent"
        elif eval_change < -100:
            quality = "🔴 Poor"
        elif eval_change < -200:
            quality = "💀 Blunder"
        else:
            quality = "🟡 Okay"
        
        print(f"📊 Move Analysis: {str(last_move)} {quality}")
        print(f"   Eval change: {eval_change:+.0f} points")
        print(f"   Position eval: {eval_after:.0f}")
        
        # Undo the temporary move
        ctx.board.pop()
        
    except Exception as e:
        print(f"⚠️ Move analysis error: {e}")


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass
