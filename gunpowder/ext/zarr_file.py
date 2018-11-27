import zarr

class ZarrFile():
    '''To be used as a context manager, similar to h5py.File.'''

    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        if mode != 'r':
            self.sync = zarr.ProcessSynchronizer("synchronizers/%s.sync"%filename)


    def __enter__(self):
        if self.mode != 'r':
            return zarr.open(self.filename, mode=self.mode, synchronizer=self.sync)
        else:
            return zarr.open(self.filename, mode=self.mode)

    def __exit__(self, *args):
        pass
