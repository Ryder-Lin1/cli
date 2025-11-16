#!/usr/bin/env python3
"""
Train a neural network to play chess by learning from master games
This learns to predict the next move given a position
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import chess
import chess.pgn
import numpy as np
import os

# Neural Network Architecture
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        
        # Input: 8x8x12 = 768 (board position)
        # 12 channels: 6 piece types x 2 colors
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        # Output: 4096 (64 from squares x 64 to squares)
        self.fc4 = nn.Linear(128, 4096)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def board_to_tensor(board):
    """Convert chess board to neural network input"""
    # 8x8x12 tensor (12 piece types)
    tensor = np.zeros((8, 8, 12), dtype=np.float32)
    
    piece_idx = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            piece_type = piece_idx[piece.piece_type]
            
            # White pieces: channels 0-5
            # Black pieces: channels 6-11
            if piece.color == chess.WHITE:
                tensor[rank][file][piece_type] = 1
            else:
                tensor[rank][file][piece_type + 6] = 1
    
    # Flatten to 768 dimensions
    return tensor.flatten()

def move_to_index(move):
    """Convert move to index (0-4095)"""
    from_square = move.from_square
    to_square = move.to_square
    return from_square * 64 + to_square

def index_to_move(index):
    """Convert index back to move"""
    from_square = index // 64
    to_square = index % 64
    return chess.Move(from_square, to_square)

class ChessDataset(Dataset):
    """Dataset of chess positions and moves"""
    def __init__(self, pgn_file, max_games=None):
        self.positions = []
        self.moves = []
        
        print(f"Loading games from {pgn_file}...")
        
        with open(pgn_file, 'r') as f:
            game_count = 0
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                
                board = game.board()
                for move in game.mainline_moves():
                    # Store position before move
                    position = board_to_tensor(board)
                    move_idx = move_to_index(move)
                    
                    self.positions.append(position)
                    self.moves.append(move_idx)
                    
                    board.push(move)
                
                game_count += 1
                if game_count % 1000 == 0:
                    print(f"  Loaded {game_count} games, {len(self.positions)} positions...")
                
                if max_games and game_count >= max_games:
                    break
        
        print(f"✅ Dataset ready: {len(self.positions)} positions from {game_count} games")
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.positions[idx]),
            torch.LongTensor([self.moves[idx]])
        )

def train_model(pgn_file, epochs=10, batch_size=256, max_games=10000):
    """Train the neural network"""
    
    # Check if CUDA available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = ChessDataset(pgn_file, max_games=max_games)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = ChessNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\n🎓 Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (positions, moves) in enumerate(dataloader):
            positions = positions.to(device)
            moves = moves.squeeze().to(device)
            
            # Forward pass
            outputs = model(positions)
            loss = criterion(outputs, moves)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(moves).sum().item()
            total += moves.size(0)
            
            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(dataloader)
        print(f"📊 Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%\n")
    
    # Save model
    torch.save(model.state_dict(), 'chess_model.pth')
    print("✅ Model saved to chess_model.pth")
    
    return model

def predict_move(model, board):
    """Use trained model to predict next move"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    with torch.no_grad():
        position = torch.FloatTensor(board_to_tensor(board)).unsqueeze(0).to(device)
        outputs = model(position)
        
        # Get top predictions and check if legal
        _, top_indices = outputs.topk(50)  # Check top 50 moves
        
        for idx in top_indices[0]:
            move = index_to_move(idx.item())
            if move in board.legal_moves:
                return move
    
    # Fallback: random legal move
    import random
    return random.choice(list(board.legal_moves))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train neural chess bot')
    parser.add_argument('--pgn', type=str, default='../../training/data/training_games.pgn',
                        help='Path to PGN file')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--max-games', type=int, default=10000,
                        help='Maximum games to train on')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pgn):
        print(f"❌ PGN file not found: {args.pgn}")
        print("Please specify correct path with --pgn")
        exit(1)
    
    model = train_model(args.pgn, args.epochs, args.batch_size, args.max_games)
    
    print("\n🎮 Testing model on starting position...")
    board = chess.Board()
    move = predict_move(model, board)
    print(f"Model suggests: {board.san(move)}")
