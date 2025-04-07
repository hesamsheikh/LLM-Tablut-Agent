import pygame
import json
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Set
from src.utils import Piece, Player, GameVisualizer, ArchiveManager, PlayerType
import asyncio

class TablutGame:
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
    
    # Add MOVE_LIMIT as a class constant
    MOVE_LIMIT = 100  # Maximum number of moves allowed

    def __init__(self):
        # Initialize 9x9 board
        self.board = [[Piece.EMPTY for _ in range(9)] for _ in range(9)]
        self.current_player = Player.WHITE
        
        # Set up initial board state
        self._setup_board()
        
        # Track board state repetitions
        self.state_count = {}
        self._update_state_count()

        self.archive_manager = ArchiveManager()
        # Add initial game state to archive
        self.archive_manager.add_game_state(self.board, None, None, None)
        self.move_callback = None  # to handle non-gui player (heuristic, llm, etc.)
        
        # Add move counter
        self.move_count = 0

    def set_move_callback(self, callback, player: Player):
        """Set a callback function to be called when it's the specified player's turn
        
        Args:
            callback: The callback function to call for moves
            player: Which player (BLACK/WHITE) this callback is for
        """
        if player == Player.BLACK:
            self.black_move_callback = callback
        else:
            self.white_move_callback = callback
        
    def notify_move_needed(self):
        """Notify when a programmatic move is needed"""
        log = None
        if self.current_player == Player.BLACK and hasattr(self, 'black_move_callback'):
            log = self.black_move_callback(self)
        elif self.current_player == Player.WHITE and hasattr(self, 'white_move_callback'):
            log = self.white_move_callback(self)
        if log:
            print(log)

    def replay_game(self, filename: str):
        """Load and replay a game from a JSON file"""
        try:
            
            game_data = self.archive_manager.load_game(filename)
            game_states = game_data['game_states']
            current_state_idx = 0
            
            # Initialize pygame if not already done
            pygame.init()
            screen = pygame.display.set_mode((801, 801))
            pygame.display.set_caption("Tablut Game Replay")
            
            # Convert notation directly using Piece enum values
            notation_to_piece = {piece.value: piece for piece in Piece}
            print(f"Loading game: {filename}")
            print("Press -> to show next move, <- for previous move")
            
            running = True
            move_from = None
            move_to = None
            
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RIGHT:
                            if current_state_idx < len(game_states) - 1:
                                current_state_idx += 1
                                move_from = game_states[current_state_idx]['move_from']
                                move_to = game_states[current_state_idx]['move_to']
                                print(f"Move {current_state_idx}: {game_states[current_state_idx]['player']} moved from "
                                      f"{move_from} to {move_to}")
                            elif current_state_idx == len(game_states) - 1:
                                print(f"\nGame Over!")
                                print(f"Winner: {game_data['winner']}")
                                print(f"Reason: {game_data['description']}")
                                print(f"Total moves: {game_data['total_moves']}")
                        elif event.key == pygame.K_LEFT and current_state_idx > 0:
                            current_state_idx -= 1
                            move_from = game_states[current_state_idx]['move_from']
                            move_to = game_states[current_state_idx]['move_to']
                            print(f"Move {current_state_idx}: {game_states[current_state_idx]['player']} moved from "
                                  f"{move_from} to {move_to}")
                
                # Update board state from game log
                current_state = game_states[current_state_idx]
                board_state = current_state['board']
                
                # Convert board state to pieces
                for row in range(9):
                    for col in range(9):
                        self.board[row][col] = notation_to_piece[board_state[row][col]]
                
                # Draw current state
                self.visualize_game_state(screen)
            
            pygame.quit()
            
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading game: {e}")
        
    def _board_to_string(self):
        """Convert current board state to string for comparison"""
        return '\n'.join(''.join(piece.value for piece in row) for row in self.board)
        
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
        """Get all valid moves for a piece at (row, col)"""
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
        
        # Check path is clear including destination
        if from_row == to_row:  # Horizontal movement
            for col in range(min(from_col, to_col), max(from_col, to_col) + 1):
                if col != from_col and self.board[from_row][col] not in [Piece.EMPTY, Piece.ESCAPE, Piece.CAMP]:
                    return False
                # King and white piece cannot pass through camps
                if piece in [Piece.WHITE, Piece.KING] and self.board[from_row][col] == Piece.CAMP:
                    return False
                # if black piece left camp, it cannot return to it
                if piece == Piece.BLACK and (from_row,from_col) not in self.CAMP_TILES and self.board[from_row][col] == Piece.CAMP:
                    return False
        else:  # Vertical movement
            for row in range(min(from_row, to_row), max(from_row, to_row) + 1):
                if row != from_row and self.board[row][from_col] not in [Piece.EMPTY, Piece.ESCAPE, Piece.CAMP]:
                    return False
                # King and white piece cannot pass through camps
                if piece in [Piece.WHITE, Piece.KING] and self.board[row][from_col] == Piece.CAMP:
                    return False
                # if black piece left camp, it cannot return to it
                if piece == Piece.BLACK and (from_row,from_col) not in self.CAMP_TILES and self.board[row][from_col] == Piece.CAMP:
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

    def move_piece(self, from_row: int, from_col: int, to_row: int, to_col: int) -> Tuple[bool, Optional[str]]:
        """Move a piece on the board and return (success, error_message)"""
        # Check if moving piece belongs to current player
        piece = self.board[from_row][from_col]
        if piece == Piece.EMPTY:
            return False, f"{self.current_player.value} tried to move an empty space"
        if self.current_player == Player.WHITE and piece not in [Piece.WHITE, Piece.KING]:
            return False, f"{self.current_player.value} tried to move opponent's piece"
        if self.current_player == Player.BLACK and piece != Piece.BLACK:
            return False, f"{self.current_player.value} tried to move opponent's piece"

        if self._is_valid_move(from_row, from_col, to_row, to_col):
            moving_player = self.current_player
            self.board[to_row][to_col] = piece
            self._clear_tile(from_row, from_col)
            
            # Increment move counter
            self.move_count += 1
            
            # Check for captures after move
            self.check_captures(to_row, to_col, moving_player)
            
            # Update state count after move
            self._update_state_count()
            
            # Add move to archive
            self.archive_manager.add_game_state(self.board, moving_player, (from_row, from_col), (to_row, to_col))
            
            # Switch turns after successful move
            self.current_player = Player.BLACK if self.current_player == Player.WHITE else Player.WHITE
            
            # Check if next player has any valid moves
            if not self._has_any_valid_moves(self.current_player):
                # Current player loses if they have no valid moves
                self.current_player = Player.BLACK if self.current_player == Player.WHITE else Player.WHITE
                return True, None
            
            # Check if game is over and save archive if it is
            if self.is_game_over():
                winner = self.get_winner()
                is_draw = winner is None
                
                # Determine reason for game end
                reason = ""
                if self.move_count >= self.MOVE_LIMIT:
                    reason = f"Draw - Move limit ({self.MOVE_LIMIT} moves) reached"
                elif self.is_king_captured():
                    reason = "King captured"
                elif self.has_king_escaped():
                    reason = "King escaped"
                elif not self._has_any_valid_moves(self.current_player):
                    reason = f"{self.current_player.value} has no valid moves"
                elif self.state_count.get(self._board_to_string(), 0) >= 3:
                    reason = "Draw - Repeated position"
                
                # self.archive_manager.save_game(winner, is_draw, description=reason)
                return True, reason  # Return the reason with success
            
            return True, None

        return False, f"{self.current_player.value} attempted invalid move from ({from_row},{from_col}) to ({to_row},{to_col})"
    
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
        """Check if the game is over by king capture, escape, draw, or move limit"""
        # Check move limit first
        if self.move_count >= self.MOVE_LIMIT:
            return True
            
        # Check for repeated state (draw)
        current_state = self._board_to_string()
        if self.state_count.get(current_state, 0) >= 3:
            return True
            
        return self.is_king_captured() or self.has_king_escaped()

    def get_winner(self):
        """Return the winner of the game if it's over, otherwise None"""
        if not self.is_game_over():
            return None
            
        # Check for move limit first
        if self.move_count >= self.MOVE_LIMIT:
            return None  # Draw due to move limit
            
        # Check for draw by repeated state
        current_state = self._board_to_string()
        if self.state_count.get(current_state, 0) >= 3:
            return None  # Draw by repetition
            
        if self.is_king_captured():
            return Player.BLACK
        return Player.WHITE

    def visualize_game_state(self, screen, selected_piece=None, valid_moves=None):
        """Draw the current game state on the screen"""
        visualizer = GameVisualizer()
        visualizer.draw_game_state(screen, self, selected_piece, valid_moves)

    


if __name__ == "__main__":
    game = TablutGame()
    visualizer = GameVisualizer()
    visualizer.run(game, white_player_type=PlayerType.GUI, black_player_type=PlayerType.GUI)
            
    ## Replay specific game
    # game_file = "Black_9_20250204_191818.json"
    # print(f"Replaying game: {game_file}")
    # game.replay_game(game_file)
