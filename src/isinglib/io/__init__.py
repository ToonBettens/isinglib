"""Reading and writing problems: JSON, NPZ and HDF5 backends behind `read`/`write`."""

from isinglib.io import hdf5, json, npz
from isinglib.io._dispatch import read, write

__all__ = ("hdf5", "json", "npz", "read", "write")
