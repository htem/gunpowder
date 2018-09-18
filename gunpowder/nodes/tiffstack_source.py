import copy
import logging
import numpy as np
import os
import json
import cv2 

from gunpowder.batch import Batch
from gunpowder.coordinate import Coordinate
import skimage.io
from gunpowder.profiling import Timing
from gunpowder.roi import Roi
from gunpowder.array import Array
from gunpowder.array_spec import ArraySpec
from .batch_provider import BatchProvider

logger = logging.getLogger(__name__)

class TiffStackSource(BatchProvider):
    '''An Tiff Stack data source.

    The offset and resolution should be specified in the metadata file for each dataset

    Args:

        filename (``string``):

            The Base Folder which contains the metadata files containing path and row column size of each dataset
            the Metadata file for each dataset should be a json file with the following information
            {
                'sections_dir':base directory of all row col directories in sections. i.e temcagt/datasets/cb2/sections
                'sections':[<LUT of z index to row_col directories>]
                'shape': [<Num Sections>,<Num Rows>, <Num Cols>]
                'tile_size': [<tile_w>,<tile_h>]
                'resolution': [z,x,y res]
                'offset': [z,x,y offset]
                'dtype': uint8
            }

        datasets (``dict``, :class:`ArrayKey` -> ``string``):

            Dictionary of array keys to metadata file names for each array this will provide

        array_specs (``dict``, :class:`ArrayKey` -> :class:`ArraySpec`, optional):

            An optional dictionary of array keys to array specs to overwrite
            the array specs automatically determined from the HDF5 file. This
            is useful to set a missing ``voxel_size``, for example. Only fields
            that are not ``None`` in the given :class:`ArraySpec` will be used.
    '''

    def __init__(
            self,
            base_path,
            datasets,
            array_specs=None):

        self.base_path = base_path
        self.datasets = datasets

        if array_specs is None:
            self.array_specs = {}
        else:
            self.array_specs = array_specs

        self.ndims = None

    def setup(self):

        for (array_key, ds_name) in self.datasets.items():
            
            spec = self.__read_spec(array_key, self.base_path, ds_name)

            self.provides(array_key, spec)

    def provide(self, request):

        timing = Timing(self)
        timing.start()

        batch = Batch()

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
                self.__read( self.datasets[array_key], dataset_roi),
                array_spec)

        logger.debug("done")

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch
    
    def __read_spec(self, array_key, base_folder, ds_name):

        js_path = os.path.join(base_folder,"%s.json"%ds_name)
        if not os.path.exists(js_path):
            raise RuntimeError("%s.json not in %s"%(ds_name, base_folder))

        dataset_spec = json.load(open(js_path,'r'))
        self.sec_dir = dataset_spec['sections_dir']
        self.sections = dataset_spec['sections']

        dims = np.asarray(dataset_spec['shape'])
        self.tile_shape = np.asarray(dataset_spec['tile_size'])

        # last two dims are assumed to be rows x columns
        dims[-2:]*=self.tile_shape
    
        if self.ndims is None:
            self.ndims = len(dims)
        else:
            assert self.ndims == len(dims)

        if array_key in self.array_specs:
            spec = self.array_specs[array_key].copy()
        else:
            spec = ArraySpec()

        if spec.voxel_size is None:
            if 'resolution' in dataset_spec:
                spec.voxel_size = Coordinate(dataset_spec['resolution'])
            else:
                spec.voxel_size = Coordinate((1,)*self.ndims)
                logger.warning("WARNING: File %s does not contain resolution information "
                               "for %s (dataset %s), voxel size has been set to %s. This "
                               "might not be what you want.",
                               self.filename, array_key, ds_name, spec.voxel_size)

        if spec.roi is None:

            if 'offset' in dataset_spec:
                offset = Coordinate(dataset_spec['offset'])
            else:
                offset = Coordinate((0,)*self.ndims)

            spec.roi = Roi(offset, dims*spec.voxel_size)

        if spec.dtype is not None:
            assert spec.dtype == dataset.dtype, ("dtype %s provided in array_specs for %s, "
                                                 "but differs from dataset %s dtype %s"%
                                                 (self.array_specs[array_key].dtype,
                                                  array_key, ds_name, dataset_spec['dtype']))
        else:
            spec.dtype = dataset_spec['dtype']

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

    def __read(self, ds_name, roi):
        arr = np.zeros(roi.get_shape())
        bounding_box = roi.get_bounding_box()
        left = bounding_box[2].start
        right = bounding_box[2].stop
        top = bounding_box[1].start
        bottom = bounding_box[1].stop
        col_index = left / self.tile_shape[1] 
        while col_index*self.tile_shape[1] < right:
            row_index = top / self.tile_shape[0]
            while row_index * self.tile_shape[0] < bottom:
                for roi_z, z_index in enumerate(range(*bounding_box[0].indices(arr.shape[0]))):
                    rc_dir = os.path.join(self.sec_dir, self.sections[str(z_index)])
                    sec_fn = self.sec_dir%(self.sections[str(z_index)],col_indx+1,row_index+1)
                    if not os.path.exists(sec_fn):
                        print "%s Does not exist"%sec_fn
                        continue
                    img = cv2.imread(sec_fn,0)
                    #image coordinates
                    i_l = col_index*self.tile_shape[1]
                    i_r = (col_index+1)*self.tile_shape[1]
                    i_t = row_index*self.tile_shape[0]
                    i_b = (row_index+1)*self.tile_shape[0]

                    t_l = max(i_l, left)
                    t_r = min(i_r,right)
                    t_t = max(i_t,top)
                    t_b = min(i_b,bottom)

                    arr[roi_z, t_t-top:t_b-top,t_l-left:t_r-left] = img[t_t-i_t:t_b-i_t,t_l-i_l:t_r-i_l]
                row_index += 1
            col_index += 1
        return arr

    def __repr__(self):

        return self.filename
