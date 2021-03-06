import copy
import logging
import numpy as np

from gunpowder.batch import Batch
from gunpowder.coordinate import Coordinate
from gunpowder.ext import h5py
from gunpowder.profiling import Timing
from gunpowder.roi import Roi
from gunpowder.array import Array
from gunpowder.array_spec import ArraySpec
from .batch_provider import BatchProvider

logger = logging.getLogger(__name__)

class Hdf5Source(BatchProvider):
    '''An HDF5 data source.

    Provides arrays from HDF5 datasets for each array key given. If the
    attribute ``resolution`` is set in an HDF5 dataset, it will be used as the
    array's ``voxel_size``. If the attribute ``offset`` is set in an HDF5
    dataset, it will be used as the offset of the :class:`Roi` for this array.
    It is assumed that the offset is given in world units.

    Args:

        filename (``string``):

            The HDF5 file.

        datasets (``dict``, :class:`ArrayKey` -> ``string``):

            Dictionary of array keys to HDF5 dataset names that this source
            offers.

        array_specs (``dict``, :class:`ArrayKey` -> :class:`ArraySpec`, optional):

            An optional dictionary of array keys to array specs to overwrite
            the array specs automatically determined from the HDF5 file. This
            is useful to set a missing ``voxel_size``, for example. Only fields
            that are not ``None`` in the given :class:`ArraySpec` will be used.
    '''

    def __init__(
            self,
            filename,
            datasets,
            array_specs=None):

        self.filename = filename
        self.datasets = datasets

        if array_specs is None:
            self.array_specs = {}
        else:
            self.array_specs = array_specs

        self.ndims = None

    def setup(self):

        hdf_file = h5py.File(self.filename, 'r')

        for (array_key, ds_name) in self.datasets.items():

            if ds_name not in hdf_file:
                raise RuntimeError("%s not in %s"%(ds_name, self.filename))

            spec = self.__read_spec(array_key, hdf_file, ds_name)

            self.provides(array_key, spec)

        hdf_file.close()

    def provide(self, request):

        timing = Timing(self)
        timing.start()

        batch = Batch()

        with h5py.File(self.filename, 'r') as hdf_file:

            for (array_key, request_spec) in request.array_specs.items():

                logger.debug("Reading %s in %s...", array_key, request_spec.roi)

                voxel_size = self.spec[array_key].voxel_size

                # scale request roi to voxel units
                dataset_roi = request_spec.roi/voxel_size

                # shift request roi into dataset
                dataset_roi = dataset_roi - self.spec[array_key].roi.get_offset()/voxel_size

                # create array spec
                array_spec = self.spec[array_key].copy()
                array_spec.roi = request_spec.roi

                # add array to batch
                batch.arrays[array_key] = Array(
                    self.__read(hdf_file, self.datasets[array_key], dataset_roi),
                    array_spec)

        logger.debug("done")

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def __read_spec(self, array_key, hdf_file, ds_name):

        dataset = hdf_file[ds_name]

        dims = Coordinate(dataset.shape)

        if self.ndims is None:
            self.ndims = len(dims)
        else:
            assert self.ndims == len(dims)

        if array_key in self.array_specs:
            spec = self.array_specs[array_key].copy()
        else:
            spec = ArraySpec()

        if spec.voxel_size is None:

            if 'resolution' in dataset.attrs:
                spec.voxel_size = Coordinate(dataset.attrs['resolution'])
            else:
                spec.voxel_size = Coordinate((1,)*self.ndims)
                logger.warning("WARNING: File %s does not contain resolution information "
                               "for %s (dataset %s), voxel size has been set to %s. This "
                               "might not be what you want.",
                               self.filename, array_key, ds_name, spec.voxel_size)

        if spec.roi is None:

            if 'offset' in dataset.attrs:
                offset = Coordinate(dataset.attrs['offset'])
            else:
                offset = Coordinate((0,)*self.ndims)

            spec.roi = Roi(offset, dims*spec.voxel_size)

        if spec.dtype is not None:
            assert spec.dtype == dataset.dtype, ("dtype %s provided in array_specs for %s, "
                                                 "but differs from dataset %s dtype %s"%
                                                 (self.array_specs[array_key].dtype,
                                                  array_key, ds_name, dataset.dtype))
        else:
            spec.dtype = dataset.dtype

        if spec.interpolatable is None:

            spec.interpolatable = spec.dtype in [
                np.float,
                np.float32,
                np.float64,
                np.float128,
                np.uint8 # assuming this is not used for labels
            ]
            logger.warning("WARNING: You didn't set 'interpolatable' for %s "
                           "(dataset %s). Based on the dtype %s, it has been "
                           "set to %s. This might not be what you want.",
                           array_key, ds_name, spec.dtype,
                           spec.interpolatable)

        return spec

    def __read(self, hdf_file, ds_name, roi):
        return np.array(hdf_file[ds_name][roi.get_bounding_box()])

    def __repr__(self):

        return self.filename
