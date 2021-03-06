import logging
import numpy as np

from gunpowder.ext import tensorflow as tf
from gunpowder.nodes.generic_predict import GenericPredict
from gunpowder.array import ArrayKey, Array
from gunpowder.tensorflow.local_server import LocalServer

logger = logging.getLogger(__name__)

class Predict(GenericPredict):
    '''Tensorflow implementation of :class:`gunpowder.nodes.Predict`.

    Args:

        checkpoint (``string``):

            Basename of a tensorflow checkpoint storing the tensorflow graph
            and associated tensor values and metadata, as created by
            :class:`gunpowder.nodes.Train`, for example.

        inputs (``dict``, ``string`` -> :class:`ArrayKey`):

            Dictionary from the names of input tensors in the network to
            array keys.

        outputs (``dict``, ``string`` -> :class:`ArrayKey`):

            Dictionary from the names of output tensors in the network to array
            keys. New arrays will be generated by this node for each entry (if
            requested downstream).

        array_specs (``dict``, :class:`ArrayKey` -> :class:`ArraySpec`, optional):

            Used to set the specs of generated arrays (``outputs``). This is
            useful to set the ``voxel_size``, for example, if they differ from
            the voxel size of the input arrays. Only fields that are not
            ``None`` in the given :class:`ArraySpec` will be used.

        graph: (``string``, optional):

            An optional path to a tensorflow computation graph that should be
            used for prediction. The checkpoint is used to restore the values
            of matching variable names in the graph. Note that the graph
            specified here can differ from the one associated to the
            checkpoint.
    '''

    def __init__(
            self,
            checkpoint,
            inputs,
            outputs,
            array_specs=None,
            graph=None):

        super(Predict, self).__init__(
            inputs,
            outputs,
            array_specs,
            spawn_subprocess=False)
        self.checkpoint = checkpoint
        self.meta_graph = graph
        self.session = None
        self.graph = None

    def start(self):

        target = LocalServer.get_target()
        logger.info("Initializing tf session, connecting to %s...", target)

        self.graph = tf.Graph()
        self.session = tf.Session(
            target=target,
            graph=self.graph)

        with self.graph.as_default():
            self.__read_checkpoint()

    def predict(self, batch, request):

        logger.debug("predicting in batch %i", batch.id)

        array_outputs = self.__collect_requested_outputs(request)
        inputs = self.__collect_provided_inputs(batch)

        # compute outputs
        outputs = self.session.run(array_outputs, feed_dict=inputs)

        for array_key in array_outputs:
            spec = self.spec[array_key].copy()
            spec.roi = request[array_key].roi
            batch.arrays[array_key] = Array(
                outputs[array_key],
                spec)

        logger.debug("predicted in batch %i", batch.id)

    def stop(self):

        if self.session is not None:
            self.session.close()
            self.graph = None
            self.session = None

    def __read_checkpoint(self):

        logger.info("Reading checkpoint...")

        # read the graph associated to the checkpoint
        if self.meta_graph is None:
            saver = tf.train.import_meta_graph(
                self.checkpoint + '.meta',
                clear_devices=True)
        # read alternative, custom graph
        else:
            saver = tf.train.import_meta_graph(
                    self.meta_graph,
                    clear_devices=True)

        # restore variables from checkpoint
        saver.restore(self.session, self.checkpoint)

    def __collect_requested_outputs(self, request):

        array_outputs = {}

        for output_name, array_key in self.outputs.items():
            if array_key in request:
                array_outputs[array_key] = output_name

        return array_outputs

    def __collect_provided_inputs(self, batch):

        inputs = {}

        for input_name, input_key in self.inputs.items():
            if isinstance(input_key, ArrayKey):
                if input_key in batch.arrays:
                    inputs[input_name] = batch.arrays[input_key].data
                else:
                    logger.warn("batch does not contain %s, input %s will not "
                                "be set", input_key, input_name)
            elif isinstance(input_key, np.ndarray):
                inputs[input_name] = input_key
            elif isinstance(input_key, str):
                inputs[input_name] = getattr(batch, input_key)
            else:
                raise Exception(
                    "Unknown network input key {}, can't be given to "
                    "network".format(input_key))

        return inputs
