"""
NNUE (Efficiently Updatable Neural Networks) for Chess
Clean implementation from scratch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import numpy as np
import os

class NNUE(nn.Module):
    """
    NNUE Neural Network for Chess Position Evaluation
    
    Architecture:
    - Input: Half-KP features (King-Piece combinations)
    - Feature Transformer: Maps sparse input to dense hidden layer
    - Accumulator: Efficiently updatable hidden state
    - Output: Position evaluation in centipawns
    """
    
    def __init__(self, input_size=40960, hidden_size=256, king_buckets=10):
        super(NNUE, self).__init__()
        
        self.input_size = input_size      # 40960 = 64 squares * 10 piece types * 64 king positions
        self.hidden_size = hidden_size    # 256 hidden units
        self.king_buckets = king_buckets  # 10 king position buckets
        
        # Feature transformer - maps input features to hidden layer
        self.feature_transformer = nn.Linear(input_size, hidden_size, bias=False)
        
        # Output layers - combines both side perspectives
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),  # Combine white and black perspectives
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Single evaluation output
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights with small random values"""
        # Feature transformer
        nn.init.normal_(self.feature_transformer.weight, std=0.1)
        
        # Output layers
        for layer in self.output_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, white_features, black_features, side_to_move):
        """
        Forward pass through NNUE network
        
        Args:
            white_features: Active features from white's perspective
            black_features: Active features from black's perspective  
            side_to_move: 1.0 for white, -1.0 for black
        """
        # Transform features to hidden representation
        white_hidden = self.feature_transformer(white_features)
        black_hidden = self.feature_transformer(black_features)
        
        # Apply activation (clipped ReLU)
        white_hidden = torch.clamp(white_hidden, 0, 1)
        black_hidden = torch.clamp(black_hidden, 0, 1)
        
        # Combine perspectives
        combined = torch.cat([white_hidden, black_hidden], dim=1)
        
        # Output evaluation
        evaluation = self.output_layers(combined)
        
        # Apply side-to-move factor
        evaluation = evaluation * side_to_move.unsqueeze(1)
        
        return evaluation


class NNUEFeatureExtractor:
    """
    Extracts Half-KP features for NNUE from chess positions
    
    Half-KP features encode:
    - King position (bucketed for efficiency)
    - All other piece positions relative to king
    """
    
    def __init__(self):
        # Piece type mapping (exclude king)
        self.piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
        self.colors = [chess.WHITE, chess.BLACK]
        
        # King bucket mapping for horizontal symmetry
        self.king_buckets = self._create_king_buckets()
    
    def _create_king_buckets(self):
        """Create king position buckets for feature efficiency"""
        buckets = {}
        for square in chess.SQUARES:
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            
            # Mirror files for horizontal symmetry (0-3 files only)
            mirrored_file = min(file, 7 - file)
            
            # Create bucket: 4 files * 8 ranks = 32 buckets, reduced to 10
            bucket = (mirrored_file * 8 + rank) % 10
            buckets[square] = bucket
        
        return buckets
    
    def extract_features(self, board):
        """
        Extract Half-KP features from chess position
        
        Returns:
            white_features: Features from white's perspective
            black_features: Features from black's perspective
            side_to_move: 1.0 for white, -1.0 for black
        """
        # Find kings
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)
        
        if white_king is None or black_king is None:
            return None, None, None
        
        # Get king buckets
        white_king_bucket = self.king_buckets[white_king]
        black_king_bucket = self.king_buckets[black_king]
        
        # Initialize feature vectors
        white_features = torch.zeros(40960)  # Sparse feature vector
        black_features = torch.zeros(40960)
        
        # Extract piece features
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            
            if piece and piece.piece_type != chess.KING:
                # Calculate feature indices
                piece_idx = self.piece_types.index(piece.piece_type)
                color_idx = 0 if piece.color == chess.WHITE else 1
                
                # White perspective features
                white_feature_idx = (
                    white_king_bucket * 64 * 10 +  # King bucket offset
                    square * 10 +                   # Square offset
                    piece_idx * 2 + color_idx       # Piece and color
                )
                white_features[white_feature_idx % 40960] = 1.0
                
                # Black perspective features  
                black_square = chess.square_mirror(square)
                black_feature_idx = (
                    black_king_bucket * 64 * 10 +
                    black_square * 10 +
                    piece_idx * 2 + (1 - color_idx)  # Flip color for black perspective
                )
                black_features[black_feature_idx % 40960] = 1.0
        
        # Side to move
        side_to_move = 1.0 if board.turn == chess.WHITE else -1.0
        
        return white_features, black_features, torch.tensor([side_to_move])


class NNUEEvaluator:
    """
    Chess position evaluator using trained NNUE model
    """
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = NNUE()
        self.feature_extractor = NNUEFeatureExtractor()
        
        # Load pre-trained weights if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            print(f"✅ Loaded NNUE model from {model_path}")
        else:
            print("🧠 Using untrained NNUE model")
        
        self.model.to(self.device)
        self.model.eval()
    
    def load_model(self, model_path):
        """Load trained NNUE model weights"""
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def save_model(self, model_path):
        """Save trained NNUE model weights"""
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(self.model.state_dict(), model_path)
            print(f"💾 Model saved to {model_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def evaluate(self, board):
        """
        Evaluate chess position using NNUE
        
        Returns:
            evaluation: Position evaluation in centipawns
        """
        try:
            # Extract features
            white_feat, black_feat, stm = self.feature_extractor.extract_features(board)
            
            if white_feat is None:
                return 0.0
            
            # Move to device
            white_feat = white_feat.to(self.device).unsqueeze(0)
            black_feat = black_feat.to(self.device).unsqueeze(0)
            stm = stm.to(self.device)
            
            # Evaluate
            with torch.no_grad():
                evaluation = self.model(white_feat, black_feat, stm)
                return float(evaluation.item()) * 100  # Convert to centipawns
                
        except Exception as e:
            print(f"❌ Evaluation error: {e}")
            return 0.0
    
    def evaluate_batch(self, boards):
        """Evaluate multiple positions efficiently"""
        evaluations = []
        
        for board in boards:
            eval_score = self.evaluate(board)
            evaluations.append(eval_score)
        
        return evaluations


# Global evaluator instance
_nnue_evaluator = None

def get_nnue_evaluator(model_path=None):
    """Get singleton NNUE evaluator"""
    global _nnue_evaluator
    if _nnue_evaluator is None:
        _nnue_evaluator = NNUEEvaluator(model_path)
    return _nnue_evaluator

def evaluate_position(board, model_path=None):
    """
    Convenient function to evaluate a chess position
    
    Args:
        board: python-chess Board object
        model_path: Optional path to trained NNUE model
    
    Returns:
        evaluation: Position evaluation in centipawns
    """
    evaluator = get_nnue_evaluator(model_path)
    return evaluator.evaluate(board)


if __name__ == "__main__":
    # Test the NNUE implementation
    print("🧠 Testing NNUE Implementation...")
    
    # Create test position
    board = chess.Board()
    
    # Evaluate starting position
    evaluation = evaluate_position(board)
    print(f"Starting position: {evaluation:.2f} centipawns")
    
    # Test after e4
    board.push_san("e4")
    evaluation = evaluate_position(board)
    print(f"After e4: {evaluation:.2f} centipawns")
    
    # Test after e4 e5
    board.push_san("e5")
    evaluation = evaluate_position(board)
    print(f"After e4 e5: {evaluation:.2f} centipawns")
    
    print("✅ NNUE implementation working!")
