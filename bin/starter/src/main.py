try:
    from .utils import chess_manager, GameContext
except ImportError:
    from utils import chess_manager, GameContext
from chess import Move
import random
import time
import chess.pgn
import os
from huggingface_hub import hf_hub_download

# Piece values for tactical evaluation
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}

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

def load_opening_database(filename, description):
    """Load a single opening database file"""
    try:
        opening_file = hf_hub_download(
            repo_id="RyderL/chess_master_games",
            filename=filename,
            repo_type="dataset"
        )
        
        games = []
        with open(opening_file, 'r') as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                games.append(game)
        
        print(f"Loaded {filename} for {description}")
        return games
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return []

# Load opening databases (automatically fetch all available files)
print("Loading opening databases...")

# Dynamically get all available files from Hugging Face
black_openings = []

try:
    from huggingface_hub import list_repo_files
    all_files = list_repo_files('RyderL/chess_master_games', repo_type='dataset')
    
    # Filter to get only the database files (exclude README.md and .gitattributes)
    database_files = [f for f in all_files if not f.endswith('.md') and not f.endswith('.gitattributes') and '.' not in f]
    
    print(f"Found {len(database_files)} database files: {', '.join(database_files)}")
    
    # Load all available database files
    for filename in database_files:
        print(f"Loading {filename}...")
        games = load_opening_database(filename, "Black")
        black_openings.extend(games)
        print(f"  ✅ Loaded {len(games)} games from {filename}")
        
except Exception as e:
    print(f"❌ Error fetching file list: {e}")
    print("Falling back to known good files...")
    # Fallback to known files if the dynamic fetch fails
    fallback_files = ["kid1", "kid2", "opening", "sideline"]
    for filename in fallback_files:
        try:
            print(f"Loading {filename}...")
            games = load_opening_database(filename, "Black")
            black_openings.extend(games)
            print(f"  ✅ Loaded {len(games)} games from {filename}")
        except Exception as file_error:
            print(f"  ⚠️ Failed to load {filename}: {file_error}")

print(f"Total opening games loaded: {len(black_openings)} (White: 0, Black: {len(black_openings)})")

# Check first 100 games for e4 openings
print("\n🔍 Checking first 100 games for e4 openings...")
e4_games = []
games_to_check = min(100, len(black_openings))

for i, game in enumerate(black_openings[:games_to_check]):
    try:
        # Get the first move of the game
        moves = list(game.mainline_moves())
        if moves:
            first_move = moves[0]
            # Check if first move is e4 (pawn to e4)
            if str(first_move) == 'e4':
                e4_games.append({
                    'game': game,
                    'index': i,
                    'white_player': game.headers.get('White', 'Unknown'),
                    'black_player': game.headers.get('Black', 'Unknown'),
                    'moves': ' '.join(str(move) for move in moves[:10])  # First 10 moves
                })
    except Exception as e:
        print(f"Error checking game {i}: {e}")

print(f"📊 Found {len(e4_games)} games starting with e4 out of {games_to_check} checked:")
for i, game_info in enumerate(e4_games[:10]):  # Show first 10 e4 games
    print(f"  {i+1}. {game_info['white_player']} vs {game_info['black_player']}")
    print(f"     Moves: {game_info['moves']}")

if len(e4_games) > 10:
    print(f"     ... and {len(e4_games) - 10} more e4 games")

print("♟️  Chess bot initialized with opening databases + tactical analysis")



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
    # This gets called every time the model needs to make a move
    # Return a python-chess Move object that is a legal move for the current position

    print("🎯 Making move for", "WHITE" if ctx.board.turn == chess.WHITE else "BLACK")
    print("📝 Current move stack:", [str(m) for m in ctx.board.move_stack])
    print("⏰ Move number:", len(ctx.board.move_stack) + 1)

    # Try to find a move from opening database first  
    if black_openings and len(ctx.board.move_stack) < 15:  # Use opening book for first 15 moves
        print("🔍 Searching opening database...")
        
        # Only use Black openings for now (White openings will be added later)
        if ctx.board.turn == chess.BLACK:
            current_openings = black_openings
            print(f"📖 Using Black opening database ({len(black_openings)} games)")
        else:
            print("⚪ White to move - no White openings loaded yet, using tactical fallback")
            current_openings = []
        
        candidate_moves = {}  # Track move frequencies: {move_str: {'move': move_obj, 'count': int, 'sources': []}}
        
        if current_openings:  # Only search if we have openings available
            games_checked = 0
            
            for game in current_openings:
                games_checked += 1
                    
                if games_checked % 200 == 0:
                    print(f"🔄 Checked {games_checked} games so far...")
                game_board = game.board()
                moves_match = True
                
                # Check if current game matches this opening
                for i, move in enumerate(game.mainline_moves()):
                    if i >= len(ctx.board.move_stack):
                        # Found a continuation move!
                        if move in ctx.board.legal_moves:
                            move_str = str(move)
                            if move_str not in candidate_moves:
                                candidate_moves[move_str] = {
                                    'move': move,
                                    'count': 0,
                                    'sources': []
                                }
                            candidate_moves[move_str]['count'] += 1
                            candidate_moves[move_str]['sources'].append({
                                'game': game.headers.get('White', 'Unknown'),
                                'source': game.headers.get('Event', 'Unknown')
                            })
                        break
                    elif i < len(ctx.board.move_stack):
                        if move != ctx.board.move_stack[i]:
                            moves_match = False
                            # If Black played differently, stop looking through this game immediately
                            break
                        game_board.push(move)
                
                if not moves_match:
                    continue
        
        # If we found candidate moves, filter by frequency and choose one
        if candidate_moves:
            # Filter moves that appear at least 2 times
            frequent_moves = {move_str: data for move_str, data in candidate_moves.items() if data['count'] >= 2}
            
            if frequent_moves:
                # Prefer moves that appear 3+ times, otherwise use 2+ times
                very_frequent = {move_str: data for move_str, data in frequent_moves.items() if data['count'] >= 3}
                
                if very_frequent:
                    chosen_moves = very_frequent
                    print(f"✨ Using high-frequency moves (3+ occurrences)")
                else:
                    chosen_moves = frequent_moves
                    print(f"📈 Using medium-frequency moves (2+ occurrences)")
                
                # Randomly select from filtered moves
                move_str = random.choice(list(chosen_moves.keys()))
                chosen_data = chosen_moves[move_str]
                move = chosen_data['move']
                count = chosen_data['count']
                
                # Show some source info
                sources = chosen_data['sources'][:3]  # Show first 3 sources
                source_names = [s['game'] for s in sources]
                
                print(f"🎯 Selected opening move: {ctx.board.san(move)} (appears {count} times)")
                print(f"📚 Sources: {', '.join(source_names)}")
                print(f"📊 Total moves found: {len(candidate_moves)}, frequent moves: {len(frequent_moves)}")
                ctx.logProbabilities({move: 1.0})
                
                # Analyze the opening move quality
                analyze_move_quality(ctx, move)
                
                return move
            else:
                print(f"⚠️ No moves appear frequently enough (need 2+ occurrences)")
                print(f"📊 Found {len(candidate_moves)} total moves but all appear only once")

    print("No opening move found, using tactical analysis...")
    
    # Fallback to tactical analysis
    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available (i probably lost didn't i)")

    # Check for tactical opportunities before random moves
    print("Checking for tactical opportunities...")
    
    if tactical_move := find_best_capture(ctx.board):
        print("⚔️ Tactical capture found!")
        ctx.logProbabilities({tactical_move: 1.0})
        
        # Analyze the tactical move quality
        analyze_move_quality(ctx, tactical_move)
        
        return tactical_move
    
    if threat_move := find_checks_and_threats(ctx.board):
        print("🚨 Emergency check found!")
        ctx.logProbabilities({threat_move: 1.0})
        
        # Analyze the threat move quality
        analyze_move_quality(ctx, threat_move)
        
        return threat_move
    
    # Try Local NNUE evaluation
    try:
        from nnue_local import evaluate_with_nnue
        
        print("🧠 Using Local NNUE evaluation...")
        
        # Evaluate all legal moves with NNUE
        legal_moves = list(ctx.board.legal_moves)
        if legal_moves and len(legal_moves) <= 30:  # Use NNUE for reasonable move counts
            print(f"🧠 Evaluating {len(legal_moves)} moves with Local NNUE...")
            
            move_evaluations = []
            
            for move in legal_moves:
                # Make move temporarily
                ctx.board.push(move)
                
                # Evaluate position with local NNUE
                evaluation = evaluate_with_nnue(ctx.board)
                
                # Undo move
                ctx.board.pop()
                
                move_evaluations.append((move, evaluation))
            
            # Sort by evaluation (best for current player)
            move_evaluations.sort(key=lambda x: x[1], reverse=True)
            
            # Select best move
            best_move, best_eval = move_evaluations[0]
            
            print(f"✅ Local NNUE selected: {ctx.board.san(best_move)} (eval: {best_eval:.2f})")
            print(f"📊 Top 3 moves:")
            for i, (move, eval_score) in enumerate(move_evaluations[:3]):
                print(f"  {i+1}. {ctx.board.san(move)}: {eval_score:.2f}")
            
            ctx.logProbabilities({best_move: 1.0})
            analyze_move_quality(ctx, best_move)
            return best_move
        else:
            if legal_moves:
                print(f"⚠️ Too many moves ({len(legal_moves)}) for NNUE evaluation - using fallback")
            else:
                print("⚠️ No legal moves to evaluate")
            
    except ImportError:
        print("⚠️ Local NNUE not available (PyTorch not installed)")
    except Exception as e:
        print(f"⚠️ Local NNUE error: {e}")

    # Try Modal minimax service if neural network fails
    try:
        # Import the minimax client (optional dependency)
        from minimax_client import get_minimax_client
        
        print("🧠 Trying Modal minimax service...")
        client = get_minimax_client()
        
        if client.is_available():
            minimax_move = client.get_best_move(ctx.board, depth=4)
            if minimax_move and minimax_move in ctx.board.legal_moves:
                print(f"🎯 Minimax found move: {ctx.board.san(minimax_move)}")
                ctx.logProbabilities({minimax_move: 1.0})
                
                # Analyze the minimax move quality (should be good!)
                analyze_move_quality(ctx, minimax_move)
                
                return minimax_move
            else:
                print("⚠️ Minimax returned invalid move")
        else:
            print("⚠️ Minimax service not available")
            
    except ImportError:
        print("⚠️ Minimax client not installed (optional)")
    except Exception as e:
        print(f"⚠️ Minimax service error: {e}")
    
    print("No AI services available, proceeding with random selection...")
    
    move_weights = [random.random() for _ in legal_moves]
    total_weight = sum(move_weights)
    # Normalize so probabilities sum to 1
    move_probs = {
        move: weight / total_weight
        for move, weight in zip(legal_moves, move_weights)
    }
    ctx.logProbabilities(move_probs)

    selected_move = random.choices(legal_moves, weights=move_weights, k=1)[0]
    
    # Analyze the position after making the move
    analyze_move_quality(ctx, selected_move)
    
    return selected_move

def analyze_move_quality(ctx: GameContext, move):
    """
    Analyze the quality of a move using minimax evaluation
    """
    try:
        # Import the minimax client (optional dependency)
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'training'))
        from minimax_client import get_minimax_client
        
        # Get evaluation before the move
        eval_before = 0
        try:
            client = get_minimax_client()
            if client.is_available():
                eval_before = client.evaluate_position(ctx.board)
        except:
            pass
        
        # Make the move temporarily to evaluate the resulting position
        ctx.board.push(move)
        
        # Get evaluation after the move
        eval_after = 0
        try:
            client = get_minimax_client()
            if client.is_available():
                eval_after = client.evaluate_position(ctx.board)
                
                # Calculate move quality
                last_move = ctx.board.peek()  # Get the last move played
                eval_change = eval_after - eval_before
                
                # Interpret the evaluation change
                if abs(eval_change) < 20:
                    quality = "⚪ Neutral"
                elif eval_change > 50:
                    quality = "🟢 Good"  
                elif eval_change > 100:
                    quality = "🔥 Excellent"
                elif eval_change < -50:
                    quality = "🔴 Poor"
                elif eval_change < -100:
                    quality = "💀 Blunder"
                else:
                    quality = "🟡 Okay"
                
                print(f"📊 Move Analysis: {str(last_move)} {quality}")
                print(f"   Eval change: {eval_change:+.0f} centipawns")
                print(f"   Position eval: {eval_after:.0f}")
                
        except Exception as e:
            print(f"⚠️ Move analysis failed: {e}")
        
        # Undo the temporary move
        ctx.board.pop()
        
    except ImportError:
        pass  # Minimax client not available, skip analysis
    except Exception as e:
        print(f"⚠️ Move analysis error: {e}")


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass
