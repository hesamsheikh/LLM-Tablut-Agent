import pygame
import json
from datetime import datetime
from enum import Enum
from typing import List, Tuple, Optional, Dict, Set
import random

class Piece(Enum):
    EMPTY = "."
    BLACK = "B"  # Black soldier
    WHITE = "W"  # White soldier 
    KING = "K"
    CASTLE = "C"
    ESCAPE = "*"
    CAMP = "#"  # Added camp piece type

class Player(Enum):
    BLACK = "Black"
    WHITE = "White"

class GameState:
    # Define board positions as class constants
    ESCAPE_TILES = [(0,1), (0,2), (0,6), (0,7), (1,0), (2,0), (1,8), (2,8),
                    (7,0), (6,0), (7,8), (6,8), (8,2), (8,1), (8,6), (8,7)]
    
    CAMP_TILES = [(0,3), (0,4), (0,5), (1,4),
                  (3,0), (3,8), (4,0), (4,1),
                  (4,7), (4,8), (5,0), (5,8),
                  (7,4), (8,3), (8,4), (8,5)]
    
    WHITE_POSITIONS = [(3,4), (4,3), (4,2), (5,4),
                       (2,4), (6,4), (4,5), (4,6)]
    
    BLACK_POSITIONS = [(0,3), (0,4), (0,5), (1,4),
                      (8,3), (8,4), (8,5), (7,4),
                      (3,0), (4,0), (5,0), (4,1),
                      (3,8), (4,8), (5,8), (4,7)]
    
    CASTLE_POSITION = (4,4)

    def __init__(self):
        # Initialize 9x9 board
        self.board = [[Piece.EMPTY for _ in range(9)] for _ in range(9)]
        self.current_player = Player.WHITE
        self.move_history = []
        self.board_states = {}  # For tracking repeated states
        self.game_id = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        
        # Set up initial board state
        self._setup_board()
        
        # Track board state repetitions
        self.state_count = {}
        self._update_state_count()
        
    def _board_to_string(self):
        """Convert current board state to string for comparison"""
        return ''.join(piece.value for row in self.board for piece in row)
        
    def _update_state_count(self):
        """Update the count of the current board state"""
        state = self._board_to_string()
        self.state_count[state] = self.state_count.get(state, 0) + 1
        
    def _has_any_valid_moves(self, player):
        """Check if the given player has any valid moves"""
        for row in range(9):
            for col in range(9):
                piece = self.board[row][col]
                if piece == Piece.EMPTY:
                    continue
                    
                is_player_piece = (
                    (player == Player.WHITE and (piece == Piece.WHITE or piece == Piece.KING)) or
                    (player == Player.BLACK and piece == Piece.BLACK)
                )
                
                if is_player_piece and self.get_valid_moves(row, col):
                    return True
        return False
        
    def _setup_board(self):
        # Set escape tiles (not corners)
        for row, col in self.ESCAPE_TILES:
            self.board[row][col] = Piece.ESCAPE
            
        # Set castle (center)
        self.board[self.CASTLE_POSITION[0]][self.CASTLE_POSITION[1]] = Piece.CASTLE
        
        # Set camp tiles
        for row, col in self.CAMP_TILES:
            self.board[row][col] = Piece.CAMP
        
        # Place king
        self.board[4][4] = Piece.KING
        
        # Place white soldiers
        for row, col in self.WHITE_POSITIONS:
            self.board[row][col] = Piece.WHITE
            
        # Place black soldiers
        for row, col in self.BLACK_POSITIONS:
            self.board[row][col] = Piece.BLACK

            
    def get_valid_moves(self, row: int, col: int) -> List[Tuple[int, int]]:
        if self.board[row][col] == Piece.EMPTY:
            return []
            
        valid_moves = []
        piece = self.board[row][col]
        
        # Check horizontal moves
        for new_col in range(9):
            if self._is_valid_move(row, col, row, new_col):
                valid_moves.append((row, new_col))
                
        # Check vertical moves
        for new_row in range(9):
            if self._is_valid_move(row, col, new_row, col):
                valid_moves.append((new_row, col))
                
        return valid_moves

    def _is_valid_move(self, from_row: int, from_col: int, 
                      to_row: int, to_col: int) -> bool:
        # Basic movement validation
        if from_row == to_row and from_col == to_col:
            return False
            
        if from_row != to_row and from_col != to_col:
            return False  # Must move orthogonally
            
        piece = self.board[from_row][from_col]
        destination = self.board[to_row][to_col]
        
        # Check if destination is occupied
        if destination not in [Piece.EMPTY, Piece.ESCAPE, Piece.CAMP]:
            return False
            
        # Only black pieces can move to camp tiles
        if (to_row, to_col) in self.CAMP_TILES and piece != Piece.BLACK:
            return False
            
        # No piece can move to or pass through castle
        if (to_row, to_col) == self.CASTLE_POSITION:
            return False
            
        # Check if path is blocked by pieces or castle
        if from_row == to_row:  # Horizontal move
            start_col = min(from_col, to_col) + 1
            end_col = max(from_col, to_col)
            for col in range(start_col, end_col):
                # Check if square is occupied by a piece
                if self.board[from_row][col] != Piece.EMPTY:
                    # Black pieces can pass through escape tiles and camp tiles
                    if piece == Piece.BLACK:
                        if self.board[from_row][col] not in [Piece.ESCAPE, Piece.CAMP]:
                            return False
                    else:
                        return False
                # Check if castle is in path
                if (from_row, col) == self.CASTLE_POSITION:
                    return False
        else:  # Vertical move
            start_row = min(from_row, to_row) + 1
            end_row = max(from_row, to_row)
            for row in range(start_row, end_row):
                # Check if square is occupied by a piece
                if self.board[row][from_col] != Piece.EMPTY:
                    # Black pieces can pass through escape tiles and camp tiles
                    if piece == Piece.BLACK:
                        if self.board[row][from_col] not in [Piece.ESCAPE, Piece.CAMP]:
                            return False
                    else:
                        return False
                # Check if castle is in path
                if (row, from_col) == self.CASTLE_POSITION:
                    return False
                    
        return True

    def _clear_tile(self, row: int, col: int):
        """Clear a tile by restoring its original state (empty, camp, castle, or escape)"""
        if (row, col) in self.CAMP_TILES:
            self.board[row][col] = Piece.CAMP
        elif (row, col) == self.CASTLE_POSITION:
            self.board[row][col] = Piece.CASTLE
        elif (row, col) in self.ESCAPE_TILES:
            self.board[row][col] = Piece.ESCAPE
        else:
            self.board[row][col] = Piece.EMPTY

    def move_piece(self, from_row: int, from_col: int, to_row: int, to_col: int) -> bool:
        """Move a piece on the board and return True if successful"""
        if self._is_valid_move(from_row, from_col, to_row, to_col):
            piece = self.board[from_row][from_col]
            moving_player = self.current_player
            self.board[to_row][to_col] = piece
            self._clear_tile(from_row, from_col)
            
            # Check for captures after move
            self.check_captures(to_row, to_col, moving_player)
            
            # Update state count after move
            self._update_state_count()
            
            # Switch turns after successful move
            self.current_player = Player.BLACK if self.current_player == Player.WHITE else Player.WHITE
            
            # Check if next player has any valid moves
            if not self._has_any_valid_moves(self.current_player):
                # Current player loses if they have no valid moves
                self.current_player = Player.BLACK if self.current_player == Player.WHITE else Player.WHITE
                return True
                
            return True
        return False
    
    def check_captures(self, row: int, col: int, moving_player):
        """Check and execute captures around the given position"""
        
        # Check all four directions for potential captures
        directions = [(0,1), (1,0), (0,-1), (-1,0)]  # E,S,W,N
        
        # First pass: Mark pieces for capture
        pieces_to_capture = set()
        
        # Check each direction pair for sandwiching captures
        for i in range(2):
            # Get opposite directions
            dir1, dir2 = directions[i], directions[i+2]
            
            # Check both directions for captures
            for direction in [dir1, dir2]:
                curr_row, curr_col = row, col
                potential_captures = []
                
                # Keep checking in this direction
                while True:
                    curr_row += direction[0] 
                    curr_col += direction[1]
                    
                    # Stop if out of bounds
                    if not (0 <= curr_row < 9 and 0 <= curr_col < 9):
                        break
                        
                    curr_piece = self.board[curr_row][curr_col]
                    
                    # Stop if empty
                    if curr_piece == Piece.EMPTY:
                        break

                    if moving_player == Player.BLACK and curr_piece in [Piece.KING, Piece.CASTLE]:
                        pieces_to_capture.clear()
                        break
                    
                    # If we hit our own piece and have potential captures
                    if ((moving_player == Player.WHITE and curr_piece in [Piece.WHITE, Piece.KING, Piece.CASTLE]) or
                        (moving_player == Player.BLACK and curr_piece in [Piece.BLACK, Piece.CAMP])):
                        if potential_captures:
                            pieces_to_capture.update(potential_captures)
                        break
                    
                    # Add enemy piece as potential capture
                    if ((moving_player == Player.WHITE and curr_piece == Piece.BLACK) or
                        (moving_player == Player.BLACK and curr_piece == Piece.WHITE)):
                        potential_captures.append((curr_row, curr_col))
        
                # Execute captures
                for capture_row, capture_col in pieces_to_capture:
                    self._clear_tile(capture_row, capture_col)


    def is_king_captured(self):
        """Check if the king is captured"""
        # Find king's position
        king_row, king_col = None, None
        for row in range(9):
            for col in range(9):
                if self.board[row][col] == Piece.KING:
                    king_row, king_col = row, col
                    break
            if king_row is not None:
                break

        # If no king found, it's captured
        if king_row is None:
            return True

        # King in castle - needs all 4 sides
        if (king_row, king_col) == self.CASTLE_POSITION:
            castle_row, castle_col = self.CASTLE_POSITION
            surrounding = [(castle_row-1,castle_col), (castle_row,castle_col+1), 
                         (castle_row+1,castle_col), (castle_row,castle_col-1)]  # N,E,S,W
            return all(self.board[r][c] == Piece.BLACK for r,c in surrounding)

        # King next to castle - needs 3 sides
        castle_row, castle_col = self.CASTLE_POSITION
        if abs(king_row - castle_row) + abs(king_col - castle_col) == 1:
            surrounding = []
            for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                new_row, new_col = king_row + dr, king_col + dc
                if 0 <= new_row < 9 and 0 <= new_col < 9:
                    if new_row != castle_row or new_col != castle_col:
                        surrounding.append((new_row, new_col))
            return all(self.board[r][c] == Piece.BLACK for r,c in surrounding)

        # King elsewhere - needs 2 opposite sides
        direction_pairs = [((0,1), (0,-1)), ((1,0), (-1,0))]
        for dir1, dir2 in direction_pairs:
            pos1 = (king_row + dir1[0], king_col + dir1[1])
            pos2 = (king_row + dir2[0], king_col + dir2[1])
            if (0 <= pos1[0] < 9 and 0 <= pos1[1] < 9 and
                0 <= pos2[0] < 9 and 0 <= pos2[1] < 9):
                if (self.board[pos1[0]][pos1[1]] == Piece.BLACK and
                    self.board[pos2[0]][pos2[1]] == Piece.BLACK):
                    return True
        return False

    def has_king_escaped(self):
        """Check if the king has reached an escape tile"""
        for row, col in self.ESCAPE_TILES:
            if self.board[row][col] == Piece.KING:
                return True
        return False

    def is_game_over(self):
        """Check if the game is over by king capture, escape, draw, or no moves"""
        # Check for repeated state (draw)
        current_state = self._board_to_string()
        if self.state_count.get(current_state, 0) >= 3:
            return True
            
        return self.is_king_captured() or self.has_king_escaped()

    def get_winner(self):
        """Return the winner of the game if it's over, otherwise None"""
        if not self.is_game_over():
            return None
            
        # Check for draw first
        current_state = self._board_to_string()
        if self.state_count.get(current_state, 0) >= 2:
            return None  # Draw
            
        if self.is_king_captured():
            return Player.BLACK
        return Player.WHITE

    def visualize_game_state(self, screen, selected_piece=None, valid_moves=None):
        """Draw the current game state on the screen"""
        # Define colors
        PIECE_WHITE = (188, 213, 245)  # Pure white
        PIECE_BLACK = (2, 3, 37)        # Pure black
        CAMP_TILE = (41, 95, 131)    # Light gray
        CASTLE_TILE = (121, 111, 58)  # Medium gray
        ESCAPE_TILE = (105, 177, 228)  # Gray
        KING_COLOR = (255, 215, 0)     # Gold
        SELECTED_OUTLINE = (37, 199, 158)  # Blue
        VALID_MOVE_MARKER = (23, 88, 74)  # Green
        INLINE_COLOR = (50, 50, 50)     # Dark gray
        EMPTY_TILE = (84, 136, 172)    # White
        
        # Draw board
        screen.fill(EMPTY_TILE)  # Pure white background
        # Define board dimensions and piece sizes
        BOARD_SIZE = 801
        GRID_CELLS = 9
        CELL_SIZE = BOARD_SIZE // GRID_CELLS
        PIECE_RADIUS = 30
        HIGHLIGHT_RADIUS = PIECE_RADIUS
        MOVE_MARKER_RADIUS = 10
        

        for row, col in self.BLACK_POSITIONS:
            pygame.draw.rect(screen, CAMP_TILE,
                          (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        row, col = self.CASTLE_POSITION
        pygame.draw.rect(screen, CASTLE_TILE, 
                (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            
        # Draw escape tiles
        for row, col in self.ESCAPE_TILES:
            pygame.draw.rect(screen, ESCAPE_TILE,
                          (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            
        # Draw camp tiles
        for row, col in self.CAMP_TILES:
            pygame.draw.rect(screen, CAMP_TILE,
                          (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                          
        # Draw pieces
        for row in range(GRID_CELLS):
            for col in range(GRID_CELLS):
                piece = self.board[row][col]
                center_x = col * CELL_SIZE + CELL_SIZE//2
                center_y = row * CELL_SIZE + CELL_SIZE//2
                
                # Draw pieces
                if piece != Piece.EMPTY and piece != Piece.CAMP and piece != Piece.ESCAPE and piece != Piece.CASTLE:
                    color = PIECE_BLACK  # Default black
                    if piece == Piece.WHITE:
                        color = PIECE_WHITE
                    elif piece == Piece.KING:
                        color = KING_COLOR
                    
                    pygame.draw.circle(screen, color, (center_x, center_y), PIECE_RADIUS)
        
        # Highlight selected piece and valid moves
        if selected_piece:
            row, col = selected_piece
            center_x = col * CELL_SIZE + CELL_SIZE//2
            center_y = row * CELL_SIZE + CELL_SIZE//2
            pygame.draw.circle(screen, SELECTED_OUTLINE, (center_x, center_y), HIGHLIGHT_RADIUS, 3)
            
            if valid_moves:
                for move_row, move_col in valid_moves:
                    center_x = move_col * CELL_SIZE + CELL_SIZE//2
                    center_y = move_row * CELL_SIZE + CELL_SIZE//2
                    pygame.draw.circle(screen, VALID_MOVE_MARKER, (center_x, center_y), MOVE_MARKER_RADIUS)
        
        # Draw grid lines
        for i in range(GRID_CELLS):
            pygame.draw.line(screen, INLINE_COLOR, 
                           (i * CELL_SIZE, 0), 
                           (i * CELL_SIZE, BOARD_SIZE))
            pygame.draw.line(screen, INLINE_COLOR, 
                           (0, i * CELL_SIZE), 
                           (BOARD_SIZE, i * CELL_SIZE))
        
        pygame.display.flip()


    def run(self):
        """Main game loop"""
        pygame.init()
        screen = pygame.display.set_mode((800, 800))
        pygame.display.set_caption("Hnefatafl")
        clock = pygame.time.Clock()
        
        running = True
        selected_piece = None
        valid_moves = []
        game_over = False
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
                if event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                    # Get board coordinates from mouse position
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    col = mouse_x // (800 // 9)
                    row = mouse_y // (800 // 9) 
                    
                    if selected_piece is None:
                        # Select piece if it belongs to current player
                        piece = self.board[row][col]
                        if piece != Piece.EMPTY:
                            is_white_turn = (self.current_player == Player.WHITE)
                            if ((piece == Piece.WHITE or piece == Piece.KING) and is_white_turn) or \
                               (piece == Piece.BLACK and not is_white_turn):
                                selected_piece = (row, col)
                                valid_moves = self.get_valid_moves(row, col)
                    else:
                        # Try to move selected piece
                        if (row, col) in valid_moves:
                            self.move_piece(selected_piece[0], selected_piece[1], row, col)
                            
                            # Check win conditions
                            if self.is_game_over():
                                winner = self.get_winner()
                                if winner == Player.WHITE:
                                    print("Game Over! White wins - King has escaped!")
                                else:
                                    print("Game Over! Black wins - King has been captured!")
                                game_over = True
                                
                        selected_piece = None
                        valid_moves = []
            
            self.visualize_game_state(screen, selected_piece, valid_moves)
            clock.tick(60)
            
        pygame.quit()

if __name__ == "__main__":
    game = GameState()
    game.run()
