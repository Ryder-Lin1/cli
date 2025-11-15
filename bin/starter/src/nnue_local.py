"""
Local NNUE (Efficiently Updatable Neural Networks) implementation
Loads pre-trained weights from local file for fast chess evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import os
import numpy as np

class NNUEModel(nn.Module):
    """Local NNUE Model for chess position evaluation"""
    
    def __init__(self):
        super().__init__()
        
        # NNUE Configuration
        self.king_buckets = 8
        self.hidden_size = 256
        self.input_size = 640  # 5 piece types * 2 colors * 64 squares
        
        # Feature transformer for each king bucket
        self.feature_transformers = nn.ModuleList([
            nn.Linear(self.input_size, self.hidden_size, bias=False)
            for _ in range(self.king_buckets)
        ])
        
        # Output layers
        self.output_transform = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 32),  # Combine both perspectives
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(), 
            nn.Linear(32, 1)
        )
        
        # Initialize with simple weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for untrained model"""
        for transformer in self.feature_transformers:
            nn.init.normal_(transformer.weight, std=0.1)
        
        for layer in self.output_transform:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, white_features, black_features, white_bucket, black_bucket, stm):
        """NNUE forward pass"""
        batch_size = len(white_features) if isinstance(white_features, list) else white_features.shape[0]
        device = next(self.parameters()).device
        
        # Convert to tensors if needed
        if isinstance(white_features, list):
            white_input = torch.zeros(batch_size, self.input_size, device=device)
            black_input = torch.zeros(batch_size, self.input_size, device=device)
            
            for i in range(batch_size):
                for feat_idx in white_features[i]:
                    if feat_idx < self.input_size:
                        white_input[i, feat_idx] = 1.0
                
                for feat_idx in black_features[i]:
                    if feat_idx < self.input_size:
                        black_input[i, feat_idx] = 1.0
        else:
            white_input = white_features
            black_input = black_features
        
        # Transform features using appropriate king buckets
        white_hidden = torch.zeros(batch_size, self.hidden_size, device=device)
        black_hidden = torch.zeros(batch_size, self.hidden_size, device=device)
        
        for i in range(batch_size):
            w_bucket = white_bucket[i] % self.king_buckets
            b_bucket = black_bucket[i] % self.king_buckets
            
            white_hidden[i] = F.relu(self.feature_transformers[w_bucket](white_input[i]))
            black_hidden[i] = F.relu(self.feature_transformers[b_bucket](black_input[i]))
        
        # Combine perspectives
        combined = torch.cat([white_hidden, black_hidden], dim=1)
        
        # Output transformation
        output = self.output_transform(combined)
        
        # Apply side-to-move perspective
        if isinstance(stm, (list, tuple)):
            stm_tensor = torch.tensor(stm, device=device, dtype=torch.float32).view(-1, 1)
        else:
            stm_tensor = stm.view(-1, 1).to(device)
        
        output = output * stm_tensor
        
        return output

class NNUEFeatures:
    """NNUE Feature extraction"""
    
    @staticmethod
    def get_king_bucket(king_square, color):
        """Get king bucket (0-7) based on king position"""
        file = chess.square_file(king_square)
        rank = chess.square_rank(king_square)
        
        # Mirror for black
        if not color:  # Black
            file = 7 - file
            rank = 7 - rank
        
        # 4x2 buckets
        bucket_file = min(file, 7 - file)  # Mirror horizontally
        bucket_rank = min(rank // 4, 1)    # Top/bottom half
        
        return bucket_file + bucket_rank * 4
    
    @staticmethod
    def board_to_nnue_features(board):
        """Convert board to NNUE half-KP features"""
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)
        
        if white_king is None or black_king is None:
            return None, None, None
        
        # Get king buckets
        white_bucket = NNUEFeatures.get_king_bucket(white_king, True)
        black_bucket = NNUEFeatures.get_king_bucket(black_king, False)
        
        # Active features for each perspective
        white_features = []
        black_features = []
        
        # Piece type mapping (excluding kings)
        piece_map = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4
        }
        
        # For each piece, create features relative to each king
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                
                # Feature indices: piece_type (0-4) + color (0/1) + square (0-63)
                piece_idx = piece_map[piece.piece_type] * 2 + (0 if piece.color else 1)
                
                # White perspective features
                white_sq = square if board.turn else chess.square_mirror(square)
                white_feature_idx = piece_idx * 64 + white_sq
                white_features.append(white_feature_idx)
                
                # Black perspective features  
                black_sq = chess.square_mirror(square) if board.turn else square
                black_feature_idx = piece_idx * 64 + black_sq
                black_features.append(black_feature_idx)
        
        return white_features, black_features, (white_bucket, black_bucket)

class LocalNNUE:
    """Local NNUE evaluation wrapper"""
    
    def __init__(self):
        self.model = None
        self.device = torch.device('cpu')  # Use CPU for faster startup
        self.model_loaded = False
        self._load_model()
    
    def _load_model(self):
        """Load NNUE model from local weights file"""
        try:
            # Create model
            self.model = NNUEModel()
            
            # Try to load weights
            weights_dir = os.path.join(os.path.dirname(__file__), "weights")
            model_path = os.path.join(weights_dir, "nnue_model.pt")
            
            if os.path.exists(model_path):
                print(f"🧠 Loading NNUE weights from {model_path}")
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model_loaded = True
                print("✅ NNUE model loaded successfully")
            else:
                print(f"⚠️ NNUE weights not found at {model_path}")
                print("🧠 Using untrained NNUE model (will return basic evaluations)")
                
                # Create weights directory if it doesn't exist
                os.makedirs(weights_dir, exist_ok=True)
                print(f"📁 Created weights directory: {weights_dir}")
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            print(f"❌ Error loading NNUE model: {e}")
            self.model = None
    
    def evaluate_position(self, board):
        """Evaluate a chess position using NNUE"""
        if self.model is None:
            return 0.0
        
        try:
            # Extract features
            white_features, black_features, buckets = NNUEFeatures.board_to_nnue_features(board)
            
            if white_features is None:
                return 0.0
            
            # Prepare input
            white_bucket = [buckets[0]]
            black_bucket = [buckets[1]] 
            stm = [1.0 if board.turn else -1.0]
            
            # Evaluate
            with torch.no_grad():
                output = self.model([white_features], [black_features], white_bucket, black_bucket, stm)
                evaluation = output.item()
            
            # Scale output to centipawns
            evaluation = evaluation * 100  # Convert to centipaws
            
            # Add some basic position understanding for untrained model
            if not self.model_loaded:
                evaluation += self._basic_evaluation(board)
            
            return evaluation
            
        except Exception as e:
            print(f"❌ NNUE evaluation error: {e}")
            return 0.0
    
    def _basic_evaluation(self, board):
        """Basic evaluation for untrained model"""
        try:
            # Simple material count
            piece_values = {
                chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
                chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0
            }
            
            material = 0
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece:
                    value = piece_values.get(piece.piece_type, 0)
                    material += value if piece.color else -value
            
            # Add some randomness to avoid repetition
            material += np.random.randint(-50, 51)
            
            return material / 100.0  # Convert to evaluation units
            
        except:
            return 0.0

# Global instance
_local_nnue = None

def get_nnue():
    """Get singleton NNUE instance"""
    global _local_nnue
    if _local_nnue is None:
        _local_nnue = LocalNNUE()
    return _local_nnue

def evaluate_with_nnue(board):
    """Evaluate position with local NNUE model"""
    nnue = get_nnue()
    return nnue.evaluate_position(board)

if __name__ == "__main__":
    # Test the local NNUE
    board = chess.Board()
    evaluation = evaluate_with_nnue(board)
    print(f"Starting position evaluation: {evaluation:.2f}")
