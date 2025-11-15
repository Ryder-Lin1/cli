"""
Simple Local NNUE Training Script
Trains NNUE model on chess positions and saves weights locally
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
    
    def __init__(self, pgn_file, max_positions=50000):
        self.positions = []
        self.max_positions = max_positions
        self._load_positions(pgn_file)
    
    def _load_positions(self, pgn_file):
        """Load chess positions from compressed PGN file"""
        print(f"📚 Loading positions from {pgn_file}...")
        
        if not os.path.exists(pgn_file):
            print(f"❌ File not found: {pgn_file}")
            return
        
        position_count = 0
        game_count = 0
        
        try:
            with open(pgn_file, 'rb') as f:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(f) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                    
                    while position_count < self.max_positions:
                        game = chess.pgn.read_game(text_stream)
                        if game is None:
                            break
                        
                        game_count += 1
                        if game_count % 100 == 0:
                            print(f"  📖 Processed {game_count} games, {position_count} positions...")
                        
                        # Extract positions from game
                        board = game.board()
                        moves = list(game.mainline_moves())
                        
                        # Skip very short games
                        if len(moves) < 10:
                            continue
                        
                        # Extract positions from different parts of the game
                        for i, move in enumerate(moves):
                            if position_count >= self.max_positions:
                                break
                            
                            # Skip opening moves (first 6) and endgame (last 10)
                            if i < 6 or i >= len(moves) - 10:
                                board.push(move)
                                continue
                            
                            # Extract features
                            white_features, black_features, buckets = NNUEFeatures.board_to_nnue_features(board)
                            
                            if white_features is not None:
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
                            
                            board.push(move)
        
        except Exception as e:
            print(f"❌ Error loading positions: {e}")
        
        print(f"✅ Loaded {len(self.positions)} positions from {game_count} games")
    
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

def train_local_nnue():
    """Train NNUE model and save weights locally"""
    
    # Find the database file
    data_dir = "/Users/jennylin/Documents/GitHub/cli/training/data"
    pgn_file = os.path.join(data_dir, "lichess_db_standard_rated_2013-07.pgn.zst")
    
    if not os.path.exists(pgn_file):
        print(f"❌ Database file not found: {pgn_file}")
        print("Please make sure the database file is in the correct location")
        return
    
    # Create dataset
    print("🎯 Creating training dataset...")
    dataset = ChessPositionDataset(pgn_file, max_positions=20000)  # Start with smaller dataset
    
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
    
    # Create model
    print("🧠 Creating NNUE model...")
    model = NNUEModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"📱 Using device: {device}")
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Training loop
    num_epochs = 10
    print(f"🚀 Training for {num_epochs} epochs...")
    
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
    
    # Save the trained model
    weights_dir = "/Users/jennylin/Documents/GitHub/cli/bin/starter/src/weights"
    os.makedirs(weights_dir, exist_ok=True)
    
    model_path = os.path.join(weights_dir, "nnue_model.pt")
    torch.save(model.state_dict(), model_path)
    
    print(f"💾 Model saved to: {model_path}")
    print("🎉 Training completed!")
    
    # Test the trained model
    print("\n🧪 Testing trained model...")
    model.eval()
    
    test_board = chess.Board()
    white_features, black_features, buckets = NNUEFeatures.board_to_nnue_features(test_board)
    
    if white_features is not None:
        with torch.no_grad():
            output = model([white_features], [black_features], [buckets[0]], [buckets[1]], torch.tensor([1.0]))
            evaluation = output.item() * 100  # Convert to centipawns
            print(f"🎯 Starting position evaluation: {evaluation:.2f} centipawns")
    
    print("✅ Local NNUE training complete!")

if __name__ == "__main__":
    train_local_nnue()
