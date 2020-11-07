import chess
import numpy as np
from typing import List, Optional
from physics.game import Action, AbstractGame


class Chess(AbstractGame) :

    def __init__(self, discount: float, board: Optional[chess.Board]=None, is_white: Optional[bool]=None):
        super().__init__(discount)
        if board is None: board = chess.Board()
        self.board = board
        self.root_board = board.copy()
        self.actions = self.legal_actions()
        self.done = False
        self.num_steps = 0

        if is_white is None : is_white = board.turn
        self.is_white = is_white

        self.observations = [self.create_image(board)]

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

    def legal_actions(self) -> List[Action]:
        return list(map(lambda i: Action(i), range(len(list(self.board.legal_moves)))))

    @property
    def action_space_size(self) -> int:
        """return action space size"""
        return len(list(self.board.legal_moves))

    @property
    def num_moves(self) -> int:
        return self.board.fullmove_number

    def step(self, action) -> float:
        """execute one step of the game conditioned by the given action"""
        if self.board.is_game_over :
            self.done = True
            return 0.
        else : self.done = False

        move = list(self.board.legal_moves)[action.index]

        next_board = self.board.copy()
        next_board.push(move)

        observation = self.create_image(next_board)
        self.observations += observation,self.is_white
        reward = self.get_reward(self.board, next_board)

        self.rewards.append(reward)

        self.rewards += reward

        self.board = next_board

        if self.board.is_game_over :
            self.done = True
            self.num_steps += 1

        return reward

    def create_image(self, board: chess.Board) :
        fen = board.copy().fen().split(' ')[0].split('/')
        obervations = np.empty(shape=(8,8), dtype=str) # make empty container
        for i,row in enumerate(fen) :
            j = 0
            for square in row :
                if square.isnumeric() :
                    for _ in range(j, int(square) + j) :
                        obervations[i][j] = '.'
                        j += 1
                    continue
                obervations[i][j] = square
                j += 1
        return obervations.tolist()

    def make_image(self, index:int):
        return self.observations[index]

    def get_reward(self, board: chess.Board, next_board: chess.Board):
        if next_board.is_checkmate() : return 100
        if next_board.is_stalemate() : return 0

        # get which side the bot was on when it moved
        white_multiplier = 1 if self.is_white else -1

        value_change = self.get_board_values(next_board, white_multiplier) - \
                       self.get_board_values(board, white_multiplier)

        return value_change

    def terminal(self) -> bool:
        over = self.board.is_game_over()
        if over : self.board = self.root_board.copy()
        return over

    def reset(self):
        self.board.reset()
        # self.rewards = []

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