import os
from typing import Any
import pandas as pd
import dask.dataframe as dd


class DFAdapter:
    def __init__(self, parallel: bool = os.getenv('PARALLEL', False)) -> None:
        if parallel:
            self.accessor = dd
        else:
            self.accessor = pd

    def __getattribute__(self, attr) -> Any:
        try:
            return super(DFAdapter, self).__getattribute__(attr)
        except AttributeError:
            return self.__dict__["accessor"].__dict__[attr]

dfa = DFAdapter()

"""Notes for writeup

1. Having issues with OOO memory on the linux box due to large trading data frame.
2. This adapter allows for loading a dask / pd dataframe handler at runtime with minimal impact to the code
    - i.e. mimimal changes required to existing code
"""