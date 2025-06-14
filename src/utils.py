import pygame
import json
from datetime import datetime
from enum import Enum
from typing import List, Tuple, Optional, Dict, Set
import random, os

class Piece(Enum):
    EMPTY = "."
    BLACK = "B"  # black soldier
    WHITE = "W"  # white soldier 
    KING = "K"
    CASTLE = "C"
    ESCAPE = "*"
    CAMP = "#"  # camp piece type

class Player(Enum):
    BLACK = "Black"
    WHITE = "White"

class PlayerType(Enum):
    GUI = "gui"
    HEURISTIC = "heuristic" 
    LLM = "llm"
    RL = "reinforcement_learning"  # for RL player type

class GameVisualizer:
    """handles drawing the game separate from the actual game logic"""
    
    def __init__(self):
        # color scheme for the pieces and board
        self.PIECE_WHITE = (188, 213, 245)  # light blue
        self.PIECE_BLACK = (2, 3, 37)       # navy blue
        self.CAMP_TILE = (41, 95, 131)      # steel blue
        self.CASTLE_TILE = (121, 111, 58)   # olive
        self.ESCAPE_TILE = (202, 101, 143)  # rose pink
        self.KING_COLOR = (255, 215, 0)     # gold
        self.SELECTED_OUTLINE = (37, 199, 158)  # turquoise
        self.VALID_MOVE_MARKER = (23, 88, 74)  # forest green
        self.INLINE_COLOR = (50, 50, 50)    # charcoal
        self.EMPTY_TILE = (84, 136, 172)    # sky blue
        
        # board size and piece dimensions
        self.BOARD_SIZE = 702
        self.GRID_CELLS = 9
        self.CELL_SIZE = self.BOARD_SIZE // self.GRID_CELLS
        self.PIECE_RADIUS = 30
        self.HIGHLIGHT_RADIUS = self.PIECE_RADIUS
        self.MOVE_MARKER_RADIUS = 10
        
        # colors for game over messages
        self.TEXT_COLOR = (255, 255, 255)  # white text
        self.GAME_OVER_BG = (0, 0, 0)      # black background

    def draw_game_state(self, screen, game_state, selected_piece=None, valid_moves=None):
        """draw the current state of the game on screen"""
        screen.fill(self.EMPTY_TILE)

        # draw the special board tiles first
        for row, col in game_state.BLACK_POSITIONS:
            pygame.draw.rect(screen, self.CAMP_TILE,
                          (col * self.CELL_SIZE, row * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))
        
        row, col = game_state.CASTLE_POSITION
        pygame.draw.rect(screen, self.CASTLE_TILE, 
                (col * self.CELL_SIZE, row * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))
            
        for row, col in game_state.ESCAPE_TILES:
            pygame.draw.rect(screen, self.ESCAPE_TILE,
                          (col * self.CELL_SIZE, row * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))
            
        for row, col in game_state.CAMP_TILES:
            pygame.draw.rect(screen, self.CAMP_TILE,
                          (col * self.CELL_SIZE, row * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))
                          
        # now draw all the pieces
        for row in range(self.GRID_CELLS):
            for col in range(self.GRID_CELLS):
                piece = game_state.board[row][col]
                center_x = col * self.CELL_SIZE + self.CELL_SIZE//2
                center_y = row * self.CELL_SIZE + self.CELL_SIZE//2
                
                if piece != Piece.EMPTY and piece != Piece.CAMP and piece != Piece.ESCAPE and piece != Piece.CASTLE:
                    color = self.PIECE_BLACK  # default to black
                    if piece == Piece.WHITE:
                        color = self.PIECE_WHITE
                    elif piece == Piece.KING:
                        color = self.KING_COLOR
                    
                    pygame.draw.circle(screen, color, (center_x, center_y), self.PIECE_RADIUS)
        
        # highlight selected piece and show valid moves
        if selected_piece:
            row, col = selected_piece
            center_x = col * self.CELL_SIZE + self.CELL_SIZE//2
            center_y = row * self.CELL_SIZE + self.CELL_SIZE//2
            pygame.draw.circle(screen, self.SELECTED_OUTLINE, (center_x, center_y), self.HIGHLIGHT_RADIUS, 3)
            
            if valid_moves:
                for move_row, move_col in valid_moves:
                    center_x = move_col * self.CELL_SIZE + self.CELL_SIZE//2
                    center_y = move_row * self.CELL_SIZE + self.CELL_SIZE//2
                    pygame.draw.circle(screen, self.VALID_MOVE_MARKER, (center_x, center_y), self.MOVE_MARKER_RADIUS)
        
        # draw the grid lines over everything
        for i in range(self.GRID_CELLS):
            pygame.draw.line(screen, self.INLINE_COLOR, 
                           (i * self.CELL_SIZE, 0), 
                           (i * self.CELL_SIZE, self.BOARD_SIZE))
            pygame.draw.line(screen, self.INLINE_COLOR, 
                           (0, i * self.CELL_SIZE), 
                           (self.BOARD_SIZE, i * self.CELL_SIZE))


    def run(self, game_state, white_player_type=PlayerType.GUI, black_player_type=PlayerType.GUI, is_visualization=False):
        """main game loop - handles both GUI and AI players"""
        pygame.init()
        WINDOW_SIZE = self.BOARD_SIZE
        screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Tablut")
        clock = pygame.time.Clock()
        
        # set up font for game over message
        font = pygame.font.Font(None, 36)

        running = True
        selected_piece = None
        valid_moves = []
        game_over = False
        game_over_reason = None
        
        while running:
            current_player_gui = (white_player_type == PlayerType.GUI if game_state.current_player == Player.WHITE 
                               else black_player_type == PlayerType.GUI)
            
            # handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
                # only process mouse clicks if current player uses GUI and game isn't over
                if event.type == pygame.MOUSEBUTTONDOWN and not game_over and current_player_gui:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    col = mouse_x // (WINDOW_SIZE // 9)
                    row = mouse_y // (WINDOW_SIZE // 9) 
                    
                    if selected_piece is None:
                        piece = game_state.board[row][col]
                        if piece != Piece.EMPTY:
                            is_white_turn = (game_state.current_player == Player.WHITE)
                            if ((piece == Piece.WHITE or piece == Piece.KING) and is_white_turn) or \
                               (piece == Piece.BLACK and not is_white_turn):
                                selected_piece = (row, col)
                                valid_moves = game_state.get_valid_moves(row, col)
                    else:
                        if (row, col) in valid_moves:
                            success, reason = game_state.move_piece(selected_piece[0], selected_piece[1], row, col)
                            if success and reason:  # if game ended with a reason
                                game_over = True
                                game_over_reason = reason
                        selected_piece = None
                        valid_moves = []
            
            # handle AI/LLM player moves
            if not game_over and not current_player_gui:
                game_state.notify_move_needed()
                if game_state.is_game_over():
                    game_over = True
                    # figure out why the game ended
                    if game_state.move_count >= game_state.MOVE_LIMIT:
                        game_over_reason = f"Draw - Move limit ({game_state.MOVE_LIMIT} moves) reached"
                    elif game_state.is_king_captured():
                        game_over_reason = "King captured"
                    elif game_state.has_king_escaped():
                        game_over_reason = "King escaped"
                    elif not game_state._has_any_valid_moves(game_state.current_player):
                        game_over_reason = f"{game_state.current_player.value} has no valid moves"
                    elif game_state.state_count.get(game_state._board_to_string(), 0) >= 3:
                        game_over_reason = "Draw - Repeated position"
            
            # draw everything
            self.draw_game_state(screen, game_state, 
                               selected_piece if current_player_gui else None,
                               valid_moves if current_player_gui else None)
            
            # show game over message if needed
            if game_over and game_over_reason:
                # create a semi-transparent overlay
                overlay = pygame.Surface((WINDOW_SIZE, 100))
                overlay.fill(self.GAME_OVER_BG)
                overlay.set_alpha(200)
                screen.blit(overlay, (0, WINDOW_SIZE // 2 - 50))
                
                # show the game over message
                winner = game_state.get_winner()
                if winner:
                    text = f"Game Over! Winner: {winner.value}"
                else:
                    text = "Game Over! Draw!"
                text_surface = font.render(text, True, self.TEXT_COLOR)
                screen.blit(text_surface, (WINDOW_SIZE // 2 - text_surface.get_width() // 2, 
                                         WINDOW_SIZE // 2 - 40))
                
                # show reason
                reason_surface = font.render(game_over_reason, True, self.TEXT_COLOR)
                screen.blit(reason_surface, (WINDOW_SIZE // 2 - reason_surface.get_width() // 2, 
                                           WINDOW_SIZE // 2))
            
            pygame.display.flip()
            
            # frame rate control
            if is_visualization:
                clock.tick(10)  # slower for visualization
            else:
                clock.tick(60)  # faster for training/evaluation

        pygame.quit()


class ArchiveManager:
    def __init__(self):
        self.game_states = []

    def add_game_state(self, board: List[List[Piece]], player: Player, move_from: Tuple[int, int], move_to: Tuple[int, int]):
        """save a game state to our archive"""
        # convert board enum values to strings so we can save them as JSON
        board_state = [[piece.value for piece in row] for row in board]
        
        state = {
            "player": player.value if player else None,
            "move_from": move_from if move_from else None,
            "move_to": move_to if move_to else None,
            "board": board_state,
        }
        self.game_states.append(state)
    def save_game(self, winner: Optional[Player], is_draw: bool = False, description: str = ""):
        """save the complete game to a JSON file"""
        # convert any numpy int64 values to regular Python ints
        game_states = []
        for state in self.game_states:
            converted_state = {
                "player": state["player"],
                "move_from": tuple(int(x) if x is not None else None for x in state["move_from"]) if state["move_from"] else None,
                "move_to": tuple(int(x) if x is not None else None for x in state["move_to"]) if state["move_to"] else None,
                "board": [[piece for piece in row] for row in state["board"]]
            }
            game_states.append(converted_state)

        game_data = {
            "winner": winner.value if winner else "Draw" if is_draw else None,
            "description": description,
            "total_moves": int(len(self.game_states)),
            "game_date": datetime.now().isoformat(),
            "game_states": game_states
        }

        # create filename with game info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result = "draw" if is_draw else f"{winner.value}" if winner else "incomplete"
        moves = int(len(self.game_states))
        filename = f"{result}_{moves}_{timestamp}.json"

        # make sure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        # save file in logs directory
        filepath = os.path.join('logs', filename)
        with open(filepath, 'w') as f:
            json.dump(game_data, f, indent=4)
        
        return filename

    @staticmethod
    def load_game(filename: str) -> Dict:
        """load a game from a JSON file"""
        try:
            filepath = os.path.join('logs', filename)
            with open(filepath, 'r') as f:
                game_data = json.load(f)
            return game_data
        except FileNotFoundError:
            raise FileNotFoundError(f"Game archive file {filename} not found")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file {filename}")