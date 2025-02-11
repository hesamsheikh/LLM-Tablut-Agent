from tablut import TablutGame
from utils import Player, Piece, GameVisualizer, PlayerType
from typing import Tuple, List, Optional
import copy
import random
import time


def evaluate_white_board(game: TablutGame) -> float:
    """
    Evaluates the current board state from White's perspective.
    Higher scores are better for White.
    """
    score = 0.0
    
    # Check for win/loss conditions first
    if game.is_game_over():
        winner = game.get_winner()
        if winner == Player.WHITE:
            return float('inf')
        elif winner == Player.BLACK:
            return float('-inf')
    
    # Count pieces
    black_count = 0
    white_count = 0
    king_row = None
    king_col = None
    
    # Scan board to count pieces and find king
    for row in range(9):
        for col in range(9):
            piece = game.board[row][col]
            if piece == Piece.BLACK:
                black_count += 1
            elif piece == Piece.WHITE:
                white_count += 1
            elif piece == Piece.KING:
                king_row = row
                king_col = col
    
    # Reward having more pieces than black
    score += (white_count - black_count) * 10
    
    # If king found, reward being near escape tiles
    if king_row is not None:
        for escape_row, escape_col in game.ESCAPE_TILES:
            distance = abs(king_row - escape_row) + abs(king_col - escape_col)
            if distance <= 2:
                score += (3 - distance) * 15  # More points for being closer
                
    return score

def evaluate_black_board(game: TablutGame) -> float:
    """
    Evaluates the current board state from Black's perspective.
    Higher scores are better for Black.
    """
    score = 0.0
    
    # Check for win/loss conditions first
    if game.is_game_over():
        winner = game.get_winner()
        if winner == Player.BLACK:
            return float('inf')
        elif winner == Player.WHITE:
            return float('-inf')
    
    # Count pieces
    black_count = 0
    white_count = 0
    king_row = None
    king_col = None
    
    # Scan board to count pieces and find king
    for row in range(9):
        for col in range(9):
            piece = game.board[row][col]
            if piece == Piece.BLACK:
                black_count += 1
            elif piece == Piece.WHITE:
                white_count += 1
            elif piece == Piece.KING:
                king_row = row
                king_col = col
    
    # Reward having more pieces than white (increased weight)
    score += (black_count - white_count) * 20
    
    # If king found, reward surrounding it
    if king_row is not None:
        black_neighbors = 0
        for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
            new_row = king_row + dr
            new_col = king_col + dc
            if 0 <= new_row < 9 and 0 <= new_col < 9:
                if game.board[new_row][new_col] == Piece.BLACK:
                    black_neighbors += 1
        # Increased weight for surrounding king
        score += black_neighbors * 30
        
        # Check if king is near escape tiles
        for escape_row, escape_col in game.ESCAPE_TILES:
            distance = abs(king_row - escape_row) + abs(king_col - escape_col)
            if distance <= 2:
                # Heavily penalize king being near escape
                score -= (3 - distance) * 40
                
                # Greatly reward having black pieces blocking the escape tile
                if game.board[escape_row][escape_col] == Piece.BLACK:
                    score += 50
                    
        # Additional strategic considerations
        if king_row == 4 and king_col == 4:
            # Penalize letting king stay in center
            score -= 30
            
    return score

class HeuristicAgent:
    def __init__(self):
        self.MAX_DEPTH = 3  # Restore to depth 3
        self.QUIESCENCE_DEPTH = 3  # Increase quiescence depth
        self.MAX_TIME = 5  # Maximum seconds to spend on a move
        self.start_time = 0
        self.transposition_table = {}
        self.killer_moves = {}  # Change to dictionary for better move storage

    def order_moves(self, game, moves, player, depth):
        """Improved move ordering with capture prioritization and killer moves"""
        scored_moves = []
        for move in moves:
            score = 0
            
            # Prioritize captures with higher weight
            if self.is_capture_move(game, move):
                score += 2000
                
            # Prioritize killer moves for this position
            position_key = str(game.board)
            if self.killer_moves.get((position_key, depth)) == move:
                score += 1000
                
            # Prioritize center control for both players
            to_pos = move['to']
            distance_to_center = abs(to_pos[0] - 4) + abs(to_pos[1] - 4)
            if player == Player.BLACK:
                score += (8 - distance_to_center) * 20
            else:
                # For white, prioritize moves towards edges (escape routes)
                score += distance_to_center * 15
                
            scored_moves.append((score, move))
            
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return [move for score, move in scored_moves]

    def is_capture_move(self, game, move):
        """Check if a move results in a capture"""
        game_copy = copy.deepcopy(game)
        from_pos, to_pos = move['from'], move['to']
        initial_pieces = sum(row.count(Piece.BLACK) + row.count(Piece.WHITE) + row.count(Piece.KING) 
                           for row in game_copy.board)
        
        game_copy.board[to_pos[0]][to_pos[1]] = game_copy.board[from_pos[0]][from_pos[1]]
        game_copy.board[from_pos[0]][from_pos[1]] = Piece.EMPTY
        
        final_pieces = sum(row.count(Piece.BLACK) + row.count(Piece.WHITE) + row.count(Piece.KING) 
                          for row in game_copy.board)
        return initial_pieces > final_pieces
    
    def get_possible_moves(self, game: TablutGame, player: Player) -> List[dict]:
        """Get all possible moves for the given player."""
        moves = []
        for row in range(9):
            for col in range(9):
                piece = game.board[row][col]
                # Check if piece belongs to current player
                if ((player == Player.WHITE and piece in [Piece.WHITE, Piece.KING]) or 
                    (player == Player.BLACK and piece == Piece.BLACK)):
                    possible_destinations = game.get_valid_moves(row, col)
                    for dest_row, dest_col in possible_destinations:
                        moves.append({
                            'from': (row, col),
                            'to': (dest_row, dest_col)
                        })
        return moves
    
    def quiescence_search(self, game, alpha, beta, player, depth):
        """Quiescence search to handle tactical sequences"""
        if depth == 0:
            if self.initial_player == Player.WHITE:
                return evaluate_white_board(game)
            return evaluate_black_board(game)
            
        standing_pat = evaluate_white_board(game) if self.initial_player == Player.WHITE else evaluate_black_board(game)
        
        if standing_pat >= beta:
            return beta
        if alpha < standing_pat:
            alpha = standing_pat
            
        moves = self.get_possible_moves(game, player)
        capture_moves = [move for move in moves if self.is_capture_move(game, move)]
        
        for move in capture_moves:
            game_copy = copy.deepcopy(game)
            from_pos, to_pos = move['from'], move['to']
            game_copy.board[to_pos[0]][to_pos[1]] = game_copy.board[from_pos[0]][from_pos[1]]
            game_copy.board[from_pos[0]][from_pos[1]] = Piece.EMPTY
            
            next_player = Player.BLACK if player == Player.WHITE else Player.WHITE
            score = -self.quiescence_search(game_copy, -beta, -alpha, next_player, depth - 1)
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
                
        return alpha
    

    def minimax(self, game, depth, alpha, beta, player):
        """Modified minimax with time checking and quiescence search"""
        if time.time() - self.start_time > self.MAX_TIME:
            raise TimeoutError
        
        # Check for immediate win/loss
        if game.is_game_over():
            if game.winner == self.initial_player:
                return None, float('inf')
            else:
                return None, float('-inf')
                
        # Check transposition table
        board_str = str(game.board)  # Convert board to string directly
        tt_entry = self.transposition_table.get((board_str, depth, player))
        if tt_entry is not None:
            return tt_entry

        # At max depth, use quiescence search instead of direct evaluation
        if depth == 0:
            score = self.quiescence_search(game, alpha, beta, player, self.QUIESCENCE_DEPTH)
            return None, score
                
        # Get all possible moves
        moves = self.get_possible_moves(game, player)
        if not moves:
            return None, float('-inf')
        moves = self.order_moves(game, moves, player, depth)

        best_move = None
        best_score = float('-inf') if player == self.initial_player else float('inf')
        
        moves = self.get_possible_moves(game, player)
        if not moves:
            return None, float('-inf')
        moves = self.order_moves(game, moves, player, depth)

        position_key = str(game.board)
        
        for move in moves:
            game_copy = copy.deepcopy(game)
            from_pos, to_pos = move['from'], move['to']
            game_copy.board[to_pos[0]][to_pos[1]] = game_copy.board[from_pos[0]][from_pos[1]]
            game_copy.board[from_pos[0]][from_pos[1]] = Piece.EMPTY
            
            next_player = Player.BLACK if player == Player.WHITE else Player.WHITE
            _, score = self.minimax(game_copy, depth - 1, alpha, beta, next_player)
            
            if player == self.initial_player:
                if score > best_score:
                    best_score = score
                    best_move = move
                    # Store killer move if it improves alpha
                    if score > alpha:
                        self.killer_moves[(position_key, depth)] = move
                alpha = max(alpha, best_score)
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
                    # Store killer move if it improves beta
                    if score < beta:
                        self.killer_moves[(position_key, depth)] = move
                beta = min(beta, best_score)
                
            if beta <= alpha:
                break

        return best_move, best_score
        
    def make_move(self, game, player: Player):
        """Iterative deepening implementation"""
        self.initial_player = player
        self.transposition_table.clear()
        self.start_time = time.time()
        
        best_move = None
        try:
            # Iterative deepening
            for depth in range(1, self.MAX_DEPTH + 1):
                move, score = self.minimax(game, depth, float('-inf'), float('inf'), player)
                if move:
                    best_move = move
        except TimeoutError:
            pass
            
        if not best_move:
            # Fall back to basic evaluation if no move found
            moves = self.get_possible_moves(game, player)
            if moves:
                best_move = moves[0]
            else:
                return "No valid moves available"
            
        from_pos = best_move['from']
        to_pos = best_move['to']
        success, log = game.move_piece(from_pos[0], from_pos[1], to_pos[0], to_pos[1])
        
        return log

def make_heuristic_move(game):
    """Callback function for the heuristic agent to make moves"""
    agent = HeuristicAgent()
    return agent.make_move(game, game.current_player)

if __name__ == "__main__":
    game = TablutGame()
    visualizer = GameVisualizer()
    
    # Set up the heuristic agent as black player
    game.set_move_callback(make_heuristic_move, Player.BLACK)
    
    # Run game with GUI white player and heuristic black player
    visualizer.run(game, white_player_type=PlayerType.GUI, black_player_type=PlayerType.HEURISTIC)

