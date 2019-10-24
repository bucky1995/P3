def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--num_episode', type = int, default = 100000)
    parser.add_argument('--learning_rate', type = float, default = 1.5e-4)
    parser.add_argument('--sample_batch_size', type=int, default=32)
    parser.add_argument('--gamma', type=float, default=0.99)
    return parser

