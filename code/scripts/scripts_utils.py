import diffuser.utils as utils

class Parser(utils.Parser):
    domain = 'overcooked' #TODO replace with 'object_rearrangement', 'AGENT', 'mocap', 'highway', 'robot', 'overcooked'
    dataset: str = f'{domain}'
    config: str = f'config.{domain}'