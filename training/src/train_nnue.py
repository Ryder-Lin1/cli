"""
NNUE Training Script
Trains the NNUE neural network on chess game data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import chess
import chess.pgn
import os
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from nnue import NNUE, NNUEFeatureExtractor


class ChessGameDataset(Dataset):
    """
    Dataset for chess positions extracted from PGN games
    """
    
    def __init__(self, pgn_path, max_positions=100000):
        self.positions = []
        self.feature_extractor = NNUEFeatureExtractor()
        self.max_positions = max_positions
        
        if os.path.exists(pgn_path):
            self._load_positions_from_pgn(pgn_path)
        else:
            print(f"⚠️ PGN file not found: {pgn_path}")
            self._create_dummy_data()
    
    def _load_positions_from_pgn(self, pgn_path):
        """Load chess positions from PGN file"""
        print(f"📚 Loading positions from {pgn_path}...")
        
        position_count = 0
        
        try:
            with open(pgn_path, 'r') as f:
                while position_count < self.max_positions:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    
                    # Extract positions from game
                    self._extract_positions_from_game(game, position_count)
                    position_count = len(self.positions)
                    
                    if position_count % 1000 == 0:
                        print(f"  📊 Loaded {position_count} positions...")
        
        except Exception as e:
            print(f"❌ Error loading PGN: {e}")
            self._create_dummy_data()
        
        print(f"✅ Loaded {len(self.positions)} total positions")
    
    def _extract_positions_from_game(self, game, current_count):
        """Extract training positions from a single game"""
        if current_count >= self.max_positions:
            return
        
        board = game.board()
        moves = list(game.mainline_moves())
        
        # Skip very short games
        if len(moves) < 10:
            return
        
        # Get game result for evaluation targets
        result = game.headers.get('Result', '*')
        if result == '1-0':
            white_target, black_target = 1.0, -1.0
        elif result == '0-1':
            white_target, black_target = -1.0, 1.0
        else:
            white_target, black_target = 0.0, 0.0
        
        # Extract positions from middle game (skip opening and endgame)
        start_move = min(8, len(moves) // 4)
        end_move = max(len(moves) - 8, len(moves) * 3 // 4)
        
        for i, move in enumerate(moves):
            if current_count + len(self.positions) >= self.max_positions:
                break
            
            if start_move <= i < end_move:
                # Extract features
                white_feat, black_feat, stm = self.feature_extractor.extract_features(board)
                
                if white_feat is not None:
                    # Target based on game outcome and side to move
                    if stm.item() > 0:  # White to move
                        target = white_target
                    else:  # Black to move
                        target = black_target
                    
                    # Add some noise to targets
                    target += random.uniform(-0.2, 0.2)
                    target = np.clip(target, -2.0, 2.0)
                    
                    self.positions.append({
                        'white_features': white_feat,
                        'black_features': black_feat,
                        'side_to_move': stm,
                        'target': torch.tensor([target], dtype=torch.float32)
                    })
            
            board.push(move)
    
    def _create_dummy_data(self):
        """Create dummy training data for testing"""
        print("🎲 Creating dummy training data...")
        
        for _ in range(min(1000, self.max_positions)):
            # Create random position
            board = chess.Board()
            
            # Make random moves
            for _ in range(random.randint(5, 15)):
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(random.choice(moves))
            
            # Extract features
            white_feat, black_feat, stm = self.feature_extractor.extract_features(board)
            
            if white_feat is not None:
                # Random target
                target = random.uniform(-1.0, 1.0)
                
                self.positions.append({
                    'white_features': white_feat,
                    'black_features': black_feat,
                    'side_to_move': stm,
                    'target': torch.tensor([target], dtype=torch.float32)
                })
        
        print(f"✅ Created {len(self.positions)} dummy positions")
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        pos = self.positions[idx]
        return (
            pos['white_features'],
            pos['black_features'],
            pos['side_to_move'],
            pos['target']
        )


class NNUETrainer:
    """
    NNUE Training Manager
    """
    
    def __init__(self, model_save_path="../models/nnue_model.pt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_save_path = model_save_path
        
        # Create model
        self.model = NNUE()
        self.model.to(self.device)
        
        # Training components
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.scheduler = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
        print(f"🖥️ Using device: {self.device}")
    
    def setup_training(self, learning_rate=0.001, weight_decay=1e-5):
        """Setup training optimizer and scheduler"""
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        batch_count = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch in progress_bar:
            white_feat, black_feat, stm, targets = batch
            
            # Move to device
            white_feat = white_feat.to(self.device)
            black_feat = black_feat.to(self.device)
            stm = stm.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(white_feat, black_feat, stm)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            batch_count += 1
            
            # Update progress bar
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        return total_loss / batch_count
    
    def validate(self, dataloader):
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for batch in dataloader:
                white_feat, black_feat, stm, targets = batch
                
                # Move to device
                white_feat = white_feat.to(self.device)
                black_feat = black_feat.to(self.device)
                stm = stm.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(white_feat, black_feat, stm)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                batch_count += 1
        
        return total_loss / batch_count
    
    def train(self, train_dataset, val_dataset=None, epochs=20, batch_size=64):
        """Full training loop"""
        print(f"🚀 Starting NNUE training for {epochs} epochs...")
        
        # Setup training
        self.setup_training()
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True if self.device.type == 'cuda' else False
            )
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\n📊 Epoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            print(f"Training Loss: {train_loss:.6f}")
            
            # Validate
            if val_loader:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                print(f"Validation Loss: {val_loss:.6f}")
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model()
                    print("💾 Saved best model!")
            else:
                # Save model periodically
                if (epoch + 1) % 5 == 0:
                    self.save_model()
        
        print("✅ Training completed!")
        return self.train_losses, self.val_losses
    
    def save_model(self):
        """Save trained model"""
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_save_path)
    
    def plot_training_history(self):
        """Plot training and validation loss"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', color='blue')
        
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss', color='red')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('NNUE Training History')
        plt.legend()
        plt.grid(True)
        plt.savefig('../models/training_history.png')
        plt.show()


def train_nnue_model():
    """Main training function"""
    print("🧠 NNUE Training Pipeline")
    print("=" * 50)
    
    # Configuration
    PGN_PATH = "../data/games.pgn"  # Put your PGN file here
    MAX_POSITIONS = 50000
    EPOCHS = 15
    BATCH_SIZE = 128
    
    # Create dataset
    print("📚 Loading training data...")
    dataset = ChessGameDataset(PGN_PATH, max_positions=MAX_POSITIONS)
    
    if len(dataset) == 0:
        print("❌ No training data available!")
        return
    
    # Split into train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    
    # Create trainer and train
    trainer = NNUETrainer()
    train_losses, val_losses = trainer.train(
        train_dataset, val_dataset, epochs=EPOCHS, batch_size=BATCH_SIZE
    )
    
    # Plot results
    trainer.plot_training_history()
    
    print("🎉 NNUE training complete!")
    print(f"💾 Model saved to: {trainer.model_save_path}")


if __name__ == "__main__":
    train_nnue_model()
