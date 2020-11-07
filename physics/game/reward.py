# this file calculates the reward for picking up a piece on the board
import chess
import numpy as np

class Reward:

    def __init__(self):

        self.piece_value = {
            'P': 1,
            'p': -1,
            'N': 3.2,
            'n': -3.2,
            'B': 3.33,
            'b': -3.33,
            'R': 5.1,
            'r': -5.1,
            'Q': 8.8,
            'q': -8.8,
        }

    def get_board_values(self, board: chess.Board, white_multiplier: int) :
        """

            :param board: chess board
            :param white_multiplier: 1 if AI is white else -1
            :return:
            """
        # using Hans Berliner's system

        fen = board.fen()

        total = 0
        for square in fen:
            if square in self.piece_value :
                total += self.piece_value[square]

        total *= white_multiplier

        return round(total,2)

    def get_reward(self, board: chess.Board, next_board: chess.Board):
        if next_board.is_checkmate() : return 100
        if next_board.is_stalemate() : return 0

        # get which side the bot was on when it moved
        white_multiplier = 1 if board.turn else -1

        value_change = self.get_board_values(next_board, white_multiplier) - \
                       self.get_board_values(board, white_multiplier)

        return value_change

    def board_to_numpy(self, board: chess.Board) :
        fen = board.fen().split(' ')[0].split('/')
        output = np.empty(shape=(8,8), dtype=str)
        for i,row in enumerate(fen) :
            j = 0
            for square in row :
                if square.isnumeric() :
                    for _ in range(j, int(square) + j) :
                        output[i][j] = '.'
                        j += 1
                    continue
                output[i][j] = square
                j += 1
        return output

    def step(self):
        pass