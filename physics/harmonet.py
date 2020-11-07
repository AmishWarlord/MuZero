import sys
sys.path.append('../../../')

from physics.config import MuZeroConfig, make_chess_config
from physics.networks import SharedStorage
from physics.self_play.self_play import run_selfplay, run_eval
from physics.training import ReplayBuffer
from physics.training import train_network

def muzero(config: MuZeroConfig):
    """
    MuZero training is split into two independent parts: Network training and
    self-play data generation.
    These two parts only communicate by transferring the latest networks checkpoint
    from the training to the self-play, and the finished games from the self-play
    to the training.
    In contrast to the original MuZero algorithm this version doesn't works with
    multiple threads, therefore the training and self-play is done alternately.
    :param config:
    :return:
    """

    storage = SharedStorage(config.new_network(), config.uniform_network(), config.new_optimizer())
    replay_buffer = ReplayBuffer(config)

    for i in range(config.num_training_loop):
        print(f'Train Step {i}')
        score_train = run_selfplay(config, storage, replay_buffer, config.num_episodes)
        train_network(config, storage, replay_buffer, config.num_epochs)

        print("Train score:", score_train)
        print("Test score:", run_eval(config, storage, 50))
        print(f"MuZero played {config.num_episodes * (i + 1)} "
              f"episodes and trained for {config.num_epochs * (i + 1)} epochs.\n")

    return storage.latest_network()


if __name__ == '__main__':
    config = make_chess_config()
    muzero(config)