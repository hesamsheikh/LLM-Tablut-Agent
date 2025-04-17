"""
Prompts for the Tablut LLM player.
Contains system prompt for game rules and move prompt for requesting moves.
"""

SYSTEM_PROMPT = """You are playing the ancient Viking board game Tablut. You must provide valid moves in JSON format.

Game Overview:
- 9x9 board with two players: White (with King) and Black (attackers)
- White's goal: Help the King escape to any esacape tile (marked '*')
- Black's goal: Capture the King before it escapes

Board Legend:
- K: King (White's special piece)
- W: White soldiers
- B: Black soldiers
- *: Escape tiles (tiles where King can win)
- C: Castle (center tile, only King can occupy)
- #: Camp tiles (Black's starting positions)
- .: Empty space

- If you are BLACK, you can only move B pieces
- If you are WHITE, you can only move W pieces and the K (King)
- When an invalid move is reported, try a completely different piece or direction

Initial Board Layout:
<BOARD>
   0 1 2 3 4 5 6 7 8
  +-------------------+
0 |.|*|*|B|B|B|*|*|.| 
1 |*|.|.|.|B|.|.|.|*| 
2 |*|.|.|.|W|.|.|.|*| 
3 |B|.|.|.|W|.|.|.|B| 
4 |B|B|W|W|K|W|W|B|B| 
5 |B|.|.|.|W|.|.|.|B| 
6 |*|.|.|.|W|.|.|.|*| 
7 |*|.|.|.|B|.|.|.|*| 
8 |.|*|*|B|B|B|*|*|.| 
</BOARD>

Movement Rules:
1. Basic Movement:
   - All pieces move orthogonally only
   - No jumping over other pieces
   - Cannot move empty square
   - Cannot move into another piece's position

2. Special Rules:
   - Only the King can stop on the castle (C)
   - White pieces can not stop on camp tiles (#)

3. Capture Mechanics:
   - Regular pieces (W, B) are captured by being sandwiched between two enemy pieces
   - King capture rules:
     * In castle: Must be surrounded on all 4 sides by Black
     * Next to castle: Must be surrounded on 3 sides by Black
     * Elsewhere: Must be sandwiched between 2 Black pieces on opposite sides

Strategic Guidelines:
White Strategy:
- Protect the King while moving towards escape tiles (*)
- Use soldiers to clear path and prevent King from being captured
- Watch for sandwiching opportunities against Black pieces

Black Strategy:
- Cut off King's escape routes
- Try to surround the King

Response Format:
You must respond with a JSON object containing:
{
    "move": {
        "from": [row, col],
        "to": [row, col]
    },
    "reasoning": "Explain your strategic thinking, including immediate goals and long-term plans"
}

Notes:
- Coordinates are zero-based (for example, (0,0) is the top-left corner, and the king is at (4,4))
- The reasoning should explain both immediate tactical goals and longer-term strategic plans
- If your move is invalid, examine the error message carefully and choose a completely different move

Respond with your move and reasoning in JSON format.
"""

MOVE_PROMPT = """{opponent_move}
Current board state:

{board_str}

You are playing as {current_player} (Move #{move_count}).
Analyze the board carefully and respond with your move in JSON format.
"""

def format_move_prompt(board_str, current_player, move_count, opponent_move=""):
    """Format the move prompt with current game state.
    
    Args:
        board_str: Current board state as string
        current_player: Current player (WHITE/BLACK)
        move_count: Number of moves made so far
    
    Returns:
        str: Formatted move prompt
    """
    return MOVE_PROMPT.format(
        board_str=board_str,
        current_player=current_player,
        move_count=move_count,
        opponent_move=opponent_move,
    )
