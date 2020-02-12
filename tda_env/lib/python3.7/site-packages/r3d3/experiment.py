import typing


class R3D3Experiment(typing.NamedTuple):
    db_path: str
    configs: typing.Dict
    binary: str
    max_nb_processes: int
