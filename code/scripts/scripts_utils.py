import diffuser.utils as utils

class Parser(utils.Parser):
    domain = '' #TODO replace with 'object_rearrangement', 'AGENT', 'mocap', 'highway', 'robot'
    dataset: str = f'{domain}'
    config: str = f'config.{domain}'