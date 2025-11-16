"""
Continue Training NNUE - Add Stockfish Games to Existing Model
Loads existing nnue_model.pt and trains on Stockfish data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import chess
import chess.pgn
import zstandard as zstd
import io
import os
import random
import numpy as np
from nnue_local import NNUEModel, NNUEFeatures

class ChessPositionDataset(Dataset):
    """Dataset of chess positions from PGN games"""
    
    def __init__(self, pgn_file, max_positions=50000, skip_positions=0):
        self.positions = []
        self.max_positions = max_positions
        self.skip_positions = skip_positions
        self._load_positions(pgn_file)
    
    def _load_positions(self, pgn_file):
        """Load chess positions from PGN file"""
        print(f"📚 Loading positions from {pgn_file}...")
        
        if not os.path.exists(pgn_file):
            print(f"❌ File not found: {pgn_file}")
            return
        
        position_count = 0
        game_count = 0
        
        try:
            # Check if file is compressed or not
            is_compressed = pgn_file.endswith('.zst')
            
            if is_compressed:
                with open(pgn_file, 'rb') as f:
                    dctx = zstd.ZstdDecompressor()
                    with dctx.stream_reader(f) as reader:
                        text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                        position_count, game_count = self._process_pgn_stream(text_stream, position_count, game_count)
            else:
                # Regular uncompressed PGN file
                with open(pgn_file, 'r', encoding='utf-8', errors='ignore') as text_stream:
                    position_count, game_count = self._process_pgn_stream(text_stream, position_count, game_count)
        
        except Exception as e:
            print(f"❌ Error loading positions: {e}")
        
        print(f"✅ Loaded {len(self.positions)} positions from {game_count} games")
    
    def _process_pgn_stream(self, text_stream, position_count, game_count):
        """Process games from a text stream"""
        total_positions_seen = 0  # Track all positions to handle skipping
        
        while position_count < self.max_positions:
            game = chess.pgn.read_game(text_stream)
            if game is None:
                break
            
            game_count += 1
            if game_count % 100 == 0:
                print(f"  📖 Processed {game_count} games, {total_positions_seen} total positions ({position_count} loaded)...")
            
            # Extract positions from game
            board = game.board()
            moves = list(game.mainline_moves())
            
            # Skip very short games
            if len(moves) < 10:
                continue
            
            # Extract positions from different parts of the game
            for i, move in enumerate(moves):
                board.push(move)
                
                # Skip positions if needed
                if total_positions_seen < self.skip_positions:
                    total_positions_seen += 1
                    continue
                
                # Skip opening (first 8 moves) and very late endgame
                if i < 8 or i > len(moves) - 3:
                    total_positions_seen += 1
                    continue
                
                # Extract features
                white_features, black_features, buckets = NNUEFeatures.board_to_nnue_features(board)
                
                if white_features is None:
                    total_positions_seen += 1
                    continue
                
                # Get game result for evaluation target
                result = game.headers.get("Result", "*")
                if result == "1-0":
                    target_eval = 1.0
                elif result == "0-1":
                    target_eval = -1.0
                elif result == "1/2-1/2":
                    target_eval = 0.0
                else:
                    total_positions_seen += 1
                    continue
                
                # Adjust target based on side to move
                if board.turn == chess.BLACK:
                    target_eval = -target_eval
                
                # Store position
                self.positions.append({
                    'white_features': white_features,
                    'black_features': black_features,
                    'white_bucket': buckets[0],
                    'black_bucket': buckets[1],
                    'stm': 1.0 if board.turn == chess.WHITE else -1.0,
                    'target': target_eval
                })
                
                position_count += 1
                total_positions_seen += 1
                
                if position_count >= self.max_positions:
                    break
            
            if position_count >= self.max_positions:
                break
        
        return position_count, game_count
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        pos = self.positions[idx]
        return (
            pos['white_features'],
            pos['black_features'],
            pos['white_bucket'],
            pos['black_bucket'],
            torch.tensor([pos['stm']], dtype=torch.float32),
            torch.tensor([pos['target']], dtype=torch.float32)
        )

def collate_fn(batch):
    """Custom collate function for batching"""
    white_feats = [item[0] for item in batch]
    black_feats = [item[1] for item in batch]
    white_buckets = [item[2] for item in batch]
    black_buckets = [item[3] for item in batch]
    stms = torch.stack([item[4] for item in batch])
    targets = torch.stack([item[5] for item in batch])
    
    return white_feats, black_feats, white_buckets, black_buckets, stms, targets

if __name__ == "__main__":
    print("=" * 60)
    print("🔄 CONTINUING TRAINING - Adding STOCKFISH games")
    print("=" * 60)
    
    # Training configuration
    pgn_file = "/Users/jennylin/Documents/GitHub/cli/training/data/sf.pgn"
    max_positions = 500000  # Load 500K Stockfish positions
    skip_positions = 0      # Start from beginning of Stockfish data
    batch_size = 64
    
    print("🎯 Creating training dataset...")
    print(f"📊 Loading {max_positions:,} Stockfish positions...")
    
    # Create dataset
    dataset = ChessPositionDataset(
        pgn_file=pgn_file,
        max_positions=max_positions,
        skip_positions=skip_positions
    )
    
    if len(dataset) == 0:
        print("❌ No positions loaded! Check your PGN file.")
        exit(1)
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"✅ Dataset ready: {len(dataset)} positions, {len(dataloader)} batches")
    
    # Load existing model
    print("\n🧠 Loading existing NNUE model...")
    model_path = os.path.join(os.path.dirname(__file__), "weights", "nnue_model.pt")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("   Please complete initial training first!")
        exit(1)
    
    model = NNUEModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"📂 Loading weights from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print(f"✅ Existing model loaded successfully!")
    print(f"📱 Using device: {device}")
    
    # Training setup - very low learning rate for high-quality data
    optimizer = optim.Adam(model.parameters(), lr=0.0003)  # Lower LR for engine games
    criterion = nn.MSELoss()
    
    # Training loop
    num_epochs = 10  # Fewer epochs for high-quality data
    print(f"🚀 Training for {num_epochs} epochs on STOCKFISH data...")
    print("💡 This ADDS engine-level knowledge to your model!")
    print("=" * 60)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        batch_count = 0
        
        for batch_idx, (white_feat, black_feat, white_bucket, black_bucket, stm, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(white_feat, black_feat, white_bucket, black_bucket, stm.to(device))
            loss = criterion(outputs.squeeze(), targets.to(device))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            if batch_idx % 20 == 0:
                print(f"  📊 Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / batch_count
        print(f"✅ Epoch {epoch+1} completed - Average Loss: {avg_loss:.4f}")
    
    # Save the improved model
    torch.save(model.state_dict(), model_path)
    
    print("=" * 60)
    print(f"💾 Model saved to: {model_path}")
    print("🎉 Stockfish training completed!")
    print("🧠 Your model now combines human AND engine knowledge!")
    
    # Test the trained model
    print("\n🧪 Testing improved model...")
    model.eval()
    
    test_board = chess.Board()
    white_features, black_features, buckets = NNUEFeatures.board_to_nnue_features(test_board)
    
    if white_features is not None:
        with torch.no_grad():
            stm = torch.tensor([1.0], device=device)
            evaluation = model([white_features], [black_features], [buckets[0]], [buckets[1]], stm)
            score_cp = evaluation.item() * 100
            
            print(f"🎯 Starting position evaluation: {score_cp:.2f} centipawns")
    
    print("✅ Stockfish training complete!")
