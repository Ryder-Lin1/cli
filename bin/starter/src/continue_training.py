"""
Continue Training NNUE - Adds New Data to Existing Model
Loads existing nnue_model.pt and trains on NEXT 500K positions
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
                # Skip opening moves (first 6) and endgame (last 10)
                if i < 6 or i >= len(moves) - 10:
                    board.push(move)
                    continue
                
                # Extract features
                white_features, black_features, buckets = NNUEFeatures.board_to_nnue_features(board)
                
                if white_features is not None:
                    # Check if we should skip this position
                    if total_positions_seen < self.skip_positions:
                        total_positions_seen += 1
                        board.push(move)
                        continue
                    
                    # Check if we've collected enough positions
                    if position_count >= self.max_positions:
                        break
                    
                    # Simple evaluation based on game outcome
                    result = game.headers.get('Result', '*')
                    
                    if result == '1-0':  # White wins
                        target = 1.0 if board.turn else -1.0
                    elif result == '0-1':  # Black wins
                        target = -1.0 if board.turn else 1.0
                    else:  # Draw or unknown
                        target = 0.0
                    
                    # Add some positional noise
                    target += random.uniform(-0.2, 0.2)
                    
                    self.positions.append({
                        'white_features': white_features,
                        'black_features': black_features,
                        'white_bucket': buckets[0],
                        'black_bucket': buckets[1],
                        'stm': 1.0 if board.turn else -1.0,
                        'target': target
                    })
                    
                    position_count += 1
                    total_positions_seen += 1
                
                board.push(move)
        
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
            pos['stm'],
            pos['target']
        )

def collate_fn(batch):
    """Custom collate function for NNUE data"""
    white_features = [item[0] for item in batch]
    black_features = [item[1] for item in batch]
    white_buckets = [item[2] for item in batch]
    black_buckets = [item[3] for item in batch]
    stms = torch.tensor([item[4] for item in batch], dtype=torch.float32)
    targets = torch.tensor([item[5] for item in batch], dtype=torch.float32)
    
    return white_features, black_features, white_buckets, black_buckets, stms, targets

def continue_training_nnue():
    """Continue training existing NNUE model on new data"""
    
    print("🔄 CONTINUING TRAINING - Loading existing model and adding new data")
    print("=" * 60)
    
    # Find the database file
    data_dir = "/Users/jennylin/Documents/GitHub/cli/training/data"
    pgn_file = os.path.join(data_dir, "AJ-CORR-PGN-000.pgn")
    
    if not os.path.exists(pgn_file):
        print(f"❌ Database file not found: {pgn_file}")
        print("Please make sure the database file is in the correct location")
        return
    
    # Check for existing model
    weights_dir = "/Users/jennylin/Documents/GitHub/cli/bin/starter/src/weights"
    model_path = os.path.join(weights_dir, "nnue_model.pt")
    
    if not os.path.exists(model_path):
        print(f"❌ No existing model found at: {model_path}")
        print("Please train the initial model first using train_nnue_local.py")
        return
    
    # Create dataset - SKIP first 500K, load NEXT 500K
    print("🎯 Creating training dataset...")
    print("⏭️  Skipping first 500,000 positions (already trained)")
    print("📊 Loading NEXT 500,000 positions (500K-1M) to ADD to existing knowledge...")
    dataset = ChessPositionDataset(pgn_file, max_positions=500000, skip_positions=500000)
    
    if len(dataset) == 0:
        print("❌ No positions loaded - cannot train")
        return
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Avoid multiprocessing issues
    )
    
    # Load existing model
    print("🧠 Loading existing NNUE model...")
    model = NNUEModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"📂 Loading weights from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print(f"✅ Existing model loaded successfully!")
    print(f"📱 Using device: {device}")
    
    # Training setup - lower learning rate for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Half the original LR
    criterion = nn.MSELoss()
    
    # Training loop
    num_epochs = 15
    print(f"🚀 Continuing training for {num_epochs} MORE epochs on NEW data...")
    print("💡 This ADDS knowledge to your existing model, doesn't replace it!")
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
    
    # Save the continued model (overwrites with improved version)
    torch.save(model.state_dict(), model_path)
    
    print("=" * 60)
    print(f"💾 Model saved to: {model_path}")
    print("🎉 Continued training completed!")
    print("🧠 Your model now knows 1,000,000 positions total!")
    
    # Test the trained model
    print("\n🧪 Testing improved model...")
    model.eval()
    
    test_board = chess.Board()
    white_features, black_features, buckets = NNUEFeatures.board_to_nnue_features(test_board)
    
    if white_features is not None:
        with torch.no_grad():
            output = model([white_features], [black_features], [buckets[0]], [buckets[1]], torch.tensor([1.0]))
            evaluation = output.item() * 100  # Convert to centipawns
            print(f"🎯 Starting position evaluation: {evaluation:.2f} centipawns")
    
    print("✅ Model ready to use! Restart your bot to use the improved model.")

if __name__ == "__main__":
    continue_training_nnue()
