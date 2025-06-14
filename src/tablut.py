import pygame
import json
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Set
from src.utils import Piece, Player, GameVisualizer, ArchiveManager, PlayerType
import asyncio

class TablutGame:
    # these are the corner-ish positions where the king can escape
    ESCAPE_TILES = [(0,1), (0,2), (0,6), (0,7), (1,0), (2,0), (1,8), (2,8),
                    (7,0), (6,0), (7,8), (6,8), (8,2), (8,1), (8,6), (8,7)]
    
    # black's starting camp positions
    CAMP_TILES = [(0,3), (0,4), (0,5), (1,4),
                  (3,0), (3,8), (4,0), (4,1),
                  (4,7), (4,8), (5,0), (5,8),
                  (7,4), (8,3), (8,4), (8,5)]
    
    # where white soldiers start
    WHITE_POSITIONS = [(3,4), (4,3), (4,2), (5,4),
                       (2,4), (6,4), (4,5), (4,6)]
    
    # where black soldiers start
    BLACK_POSITIONS = [(0,3), (0,4), (0,5), (1,4),
                      (8,3), (8,4), (8,5), (7,4),
                      (3,0), (4,0), (5,0), (4,1),
                      (3,8), (4,8), (5,8), (4,7)]
    
    CASTLE_POSITION = (4,4)
    
    # max moves before we call it a draw
    MOVE_LIMIT = 20

    def __init__(self):
        # start with an empty 9x9 board
        self.board = [[Piece.EMPTY for _ in range(9)] for _ in range(9)]
        self.current_player = Player.WHITE
        
        # put all the pieces in their starting positions
        self._setup_board()
        
        # keep track of repeated board positions (for draw detection)
        self.state_count = {}
        self._update_state_count()

        self.archive_manager = ArchiveManager()
        # save the initial state
        self.archive_manager.add_game_state(self.board, None, None, None)
        self.move_callback = None  # for handling non-gui players (ai, llm, etc.)
        
        # track how many moves have been made
        self.move_count = 0
        
        # remember the last move each player made
        self.last_move = {
            Player.WHITE: None,
            Player.BLACK: None
        }

    def set_move_callback(self, callback, player: Player):
        """hook up a callback function for when it's a specific player's turn
        
        Args:
            callback: The function to call for moves
            player: Which player (BLACK/WHITE) this callback is for
        """
        if player == Player.BLACK:
            self.black_move_callback = callback
        else:
            self.white_move_callback = callback
        
    def notify_move_needed(self):
        """tell the appropriate player it's their turn to move"""
        log = None
        if self.current_player == Player.BLACK and hasattr(self, 'black_move_callback'):
            log = self.black_move_callback(self)
        elif self.current_player == Player.WHITE and hasattr(self, 'white_move_callback'):
            log = self.white_move_callback(self)
        if log:
            print(log)

    def replay_game(self, filename: str):
        """load and replay a saved game from a JSON file"""
        try:
            
            game_data = self.archive_manager.load_game(filename)
            game_states = game_data['game_states']
            current_state_idx = 0
            
            # fire up pygame if it's not running yet
            pygame.init()
            screen = pygame.display.set_mode((801, 801))
            pygame.display.set_caption("Tablut Game Replay")
            
            # map the notation back to actual piece types
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
                
                # update the board to match the current state
                current_state = game_states[current_state_idx]
                board_state = current_state['board']
                
                # convert the saved board back to pieces
                for row in range(9):
                    for col in range(9):
                        self.board[row][col] = notation_to_piece[board_state[row][col]]
                
                # show the current state
                self.visualize_game_state(screen)
            
            pygame.quit()
            
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading game: {e}")
        
    def _board_to_string(self):
        """convert the current board to a string for comparison purposes"""
        return '\n'.join(''.join(piece.value for piece in row) for row in self.board)
        
    def _update_state_count(self):
        """keep track of how many times we've seen this board position"""
        state = self._board_to_string()
        self.state_count[state] = self.state_count.get(state, 0) + 1
        
    def _has_any_valid_moves(self, player):
        """check if a player has any legal moves left"""
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
        # put escape tiles in their spots (not the corners though)
        for row, col in self.ESCAPE_TILES:
            self.board[row][col] = Piece.ESCAPE
            
        # castle goes in the center
        self.board[self.CASTLE_POSITION[0]][self.CASTLE_POSITION[1]] = Piece.CASTLE
        
        # set up the camp tiles
        for row, col in self.CAMP_TILES:
            self.board[row][col] = Piece.CAMP
        
        # put the king in the castle
        self.board[4][4] = Piece.KING
        
        # place white soldiers around the king
        for row, col in self.WHITE_POSITIONS:
            self.board[row][col] = Piece.WHITE
            
        # place black soldiers around the edges
        for row, col in self.BLACK_POSITIONS:
            self.board[row][col] = Piece.BLACK

            
    def get_valid_moves(self, row: int, col: int) -> List[Tuple[int, int]]:
        """find all the places this piece can legally move to"""
        if self.board[row][col] == Piece.EMPTY:
            return []
            
        valid_moves = []
        piece = self.board[row][col]
        
        # try moving horizontally
        for new_col in range(9):
            if self._is_valid_move(row, col, row, new_col):
                valid_moves.append((row, new_col))
                
        # try moving vertically
        for new_row in range(9):
            if self._is_valid_move(row, col, new_row, col):
                valid_moves.append((new_row, col))
                
        return valid_moves


    def _is_valid_move(self, from_row: int, from_col: int, 
                      to_row: int, to_col: int) -> bool:
        # basic validation first
        if from_row == to_row and from_col == to_col:
            return False
            
        if from_row != to_row and from_col != to_col:
            return False  # can only move straight lines
            
        piece = self.board[from_row][from_col]
        
        # make sure the path is clear including destination
        if from_row == to_row:  # moving horizontally
            for col in range(min(from_col, to_col), max(from_col, to_col) + 1):
                if col != from_col and self.board[from_row][col] not in [Piece.EMPTY, Piece.ESCAPE, Piece.CAMP]:
                    return False
                # king and white pieces can't go through camps
                if piece in [Piece.WHITE, Piece.KING] and self.board[from_row][col] == Piece.CAMP:
                    return False
                # if black piece left camp, it can't go back
                if piece == Piece.BLACK and (from_row,from_col) not in self.CAMP_TILES and self.board[from_row][col] == Piece.CAMP:
                    return False
        else:  # moving vertically
            for row in range(min(from_row, to_row), max(from_row, to_row) + 1):
                if row != from_row and self.board[row][from_col] not in [Piece.EMPTY, Piece.ESCAPE, Piece.CAMP]:
                    return False
                # king and white pieces can't go through camps
                if piece in [Piece.WHITE, Piece.KING] and self.board[row][from_col] == Piece.CAMP:
                    return False
                # if black piece left camp, it can't go back
                if piece == Piece.BLACK and (from_row,from_col) not in self.CAMP_TILES and self.board[row][from_col] == Piece.CAMP:
                    return False

        return True


    def _clear_tile(self, row: int, col: int):
        """reset a tile back to what it originally was (empty, camp, castle, or escape)"""
        if (row, col) in self.CAMP_TILES:
            self.board[row][col] = Piece.CAMP
        elif (row, col) == self.CASTLE_POSITION:
            self.board[row][col] = Piece.CASTLE
        elif (row, col) in self.ESCAPE_TILES:
            self.board[row][col] = Piece.ESCAPE
        else:
            self.board[row][col] = Piece.EMPTY

    def move_piece(self, from_row: int, from_col: int, to_row: int, to_col: int) -> Tuple[bool, Optional[str]]:
        """attempt to move a piece and return whether it worked"""
        # make sure the player is moving their own piece
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
            
            # bump up the move counter
            self.move_count += 1
            
            # see if we captured anything
            captured_pieces = self.check_captures(to_row, to_col, moving_player)
            
            # update our position tracking
            self._update_state_count()
            
            # save this move to the archive
            self.archive_manager.add_game_state(self.board, moving_player, (from_row, from_col), (to_row, to_col))
            
            # switch to the other player
            self.current_player = Player.BLACK if self.current_player == Player.WHITE else Player.WHITE
            
            # check if the next player can actually move
            if not self._has_any_valid_moves(self.current_player):
                # current player loses if they have no valid moves
                self.current_player = Player.BLACK if self.current_player == Player.WHITE else Player.WHITE
                return True, None
            
            # see if the game is over and archive it if so
            if self.is_game_over():
                winner = self.get_winner()
                is_draw = winner is None
                
                # figure out why the game ended
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
                return True, reason  # return the reason with success
            
            # remember what this player just did
            move_data = {
                "from": (from_row, from_col),
                "to": (to_row, to_col),
                "piece": piece,
                "move_count": self.move_count
            }
            if captured_pieces:
                move_data["capture"] = captured_pieces[0]
            self.last_move[moving_player] = move_data
            
            return True, None

        return False, f"{self.current_player.value} attempted invalid move from ({from_row},{from_col}) to ({to_row},{to_col})"
    
    def check_captures(self, row: int, col: int, moving_player):
        """check if we captured any enemy pieces with this move"""
        
        # look in all four directions for captures
        directions = [(0,1), (1,0), (0,-1), (-1,0)]  # E,S,W,N
        
        # first pass: figure out what pieces should be captured
        pieces_to_capture = set()
        
        # check both horizontal and vertical for captures
        for i in range(2):
            # get opposite directions
            dir1, dir2 = directions[i], directions[i+2]

            # check both directions for captures
            for direction in [dir1, dir2]:
                curr_row, curr_col = row, col
                potential_captures = []

                # keep looking in this direction
                while True:
                    curr_row += direction[0] 
                    curr_col += direction[1]

                    # stop if we hit the edge
                    if not (0 <= curr_row < 9 and 0 <= curr_col < 9):
                        break

                    curr_piece = self.board[curr_row][curr_col]

                    # stop if we hit empty space
                    if curr_piece == Piece.EMPTY:
                        break

                    if moving_player == Player.BLACK and curr_piece in [Piece.KING, Piece.CASTLE]:
                        potential_captures = []
                        break

                    # if we hit our own piece and have potential captures
                    if ((moving_player == Player.WHITE and curr_piece in [Piece.WHITE, Piece.KING, Piece.CASTLE]) or
                        (moving_player == Player.BLACK and curr_piece in [Piece.BLACK, Piece.CAMP])):
                        if potential_captures:
                            pieces_to_capture.update(potential_captures)
                        break

                    # add enemy piece as potential capture
                    if ((moving_player == Player.WHITE and curr_piece == Piece.BLACK) or
                        (moving_player == Player.BLACK and curr_piece == Piece.WHITE)):
                        potential_captures.append((curr_row, curr_col))
        
        # actually remove the captured pieces
        captured_list = list(pieces_to_capture)
        for capture_row, capture_col in captured_list:
            self._clear_tile(capture_row, capture_col)
        return captured_list


    def is_king_captured(self):
        """check if the king got captured"""
        # find where the king is
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

    def get_last_move(self, player: Player = None) -> Optional[dict]:
        """
        Get the last move made by the specified player.
        
        Args:
            player: The player whose move to return. If None, returns the most recent move.
        
        Returns:
            dict: Contains 'from', 'to', 'piece', and 'move_count' if a move exists, otherwise None
        """
        if player is not None:
            return self.last_move[player]
        
        # If no player specified, return the most recent move based on move_count
        white_move = self.last_move[Player.WHITE]
        black_move = self.last_move[Player.BLACK]
        
        if white_move is None and black_move is None:
            return None
        elif white_move is None:
            return black_move
        elif black_move is None:
            return white_move
        
        # Return the move with the higher move_count
        return white_move if white_move['move_count'] > black_move['move_count'] else black_move


if __name__ == "__main__":
    game = TablutGame()
    visualizer = GameVisualizer()
    visualizer.run(game, white_player_type=PlayerType.GUI, black_player_type=PlayerType.GUI)
            
    ## Replay specific game
    # game_file = "Black_9_20250204_191818.json"
    # print(f"Replaying game: {game_file}")
    # game.replay_game(game_file)
