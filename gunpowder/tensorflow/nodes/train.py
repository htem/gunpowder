import logging
import os
import numpy as np

from gunpowder.ext import tensorflow as tf
from gunpowder.nodes.generic_train import GenericTrain
from gunpowder.volume import VolumeType, Volume

logger = logging.getLogger(__name__)

class Train(GenericTrain):
    '''Tensorflow implementation of :class:`gunpowder.nodes.Train`.

    Args:

        meta_graph_filename: Filename of a tensorflow meta-graph storing the
            tensorflow graph containing an optimizer. A meta-graph file can be
            created by running::

                # create tensorflow graph
                ...

                # store it
                tf.train.export_meta_graph(filename=meta_graph_filename)

        optimizer (string or function): Either the name of the tensorflow
            operator performing a training iteration, or a function that, given
            the graph of the meta-graph file, adds a custom loss and optimizer.

            If a function is given, it should return a tuple ``(loss,
            optimizer)`` of a tensor and an operator representing the loss and
            the optimizer, respectively. In this case, parameter ``loss``
            should be ``None``.

            Example::

                def add_custom_optimizer(graph):

                    # get the output of your graph
                    output = graph.get_tensor_by_name('...')

                    # create your custom loss
                    loss = custom_loss(output)

                    # add an optimizer of your choice
                    optimizer = tf.train.AdamOptimizer().minimize(loss)

                    return (loss, optimizer)

        loss (string or None): The name of the tensorflow tensor containing the
            loss, or ``None`` if ``optimizer`` is a function.

        inputs (dict): Dictionary from the names of input tensors in the
            network to :class:``VolumeType`` or batch attribute name as string.

        outputs (dict): Dictionary from the names of output tensors in the
            network to :class:``VolumeType``. New volumes will be generated by
            this node for each entry (if requested downstream).

        gradients (dict): Dictionary from the names of output tensors in the
            network to :class:``VolumeType``. New volumes containing the
            gradient of an output with respect to the loss will be generated by
            this node for each entry (if requested downstream).

        volume_specs (dict, optional): An optional dictionary of
            :class:`VolumeType` to :class:`VolumeSpec` to set the volume specs
            of generated volumes (``outputs`` and ``gradients``). This is useful
            to set the ``voxel_size``, for example, if they differ from the
            voxel size of the input volumes. Only fields that are not ``None``
            in the given :class:`VolumeSpec` will be used.

        save_every (int, optional): After how many iterations to create a
            checkpoint to store the learnt weights.
    '''

    def __init__(
            self,
            meta_graph_filename,
            optimizer,
            loss,
            inputs,
            outputs,
            gradients,
            volume_specs=None,
            save_every=2000,
            checkpoint_dir=''):

        super(Train, self).__init__(
            inputs,
            outputs,
            gradients,
            volume_specs,
            spawn_subprocess=False)
        self.meta_graph_filename = meta_graph_filename
        self.optimizer_func = None
        self.optimizer_loss_names = None
        self.optimizer = None
        self.loss = None
        self.session = None
        self.tf_gradient = {}
        self.graph = None
        self.basic_saver = None
        self.full_saver = None
        self.save_every = save_every
        self.iteration = None
        self.iteration_increment = None
        self.checkpoint_dir = checkpoint_dir

        if isinstance(optimizer, basestring):
            self.optimizer_loss_names = (optimizer, loss)
        else:
            self.optimizer_func = optimizer

    def start(self):

        logger.info("Initializing tf session...")

        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

        with self.graph.as_default():
            self.__read_meta_graph()

        if self.optimizer_func is None:

            # get actual operations/tensors from names
            self.optimizer = self.graph.get_operation_by_name(self.optimizer_loss_names[0])
            self.loss = self.graph.get_tensor_by_name(self.optimizer_loss_names[1])

        # add symbolic gradients
        for tensor_name in self.gradients:
            tensor = self.graph.get_tensor_by_name(tensor_name)
            self.tf_gradient[tensor_name] = tf.gradients(
                self.loss,
                [tensor])[0]

    def train_step(self, batch, request):

        volume_outputs = self.__collect_requested_outputs(request)
        inputs = self.__collect_provided_inputs(batch)

        to_compute = {
            'optimizer': self.optimizer,
            'loss': self.loss,
            'iteration': self.iteration_increment}
        to_compute.update(volume_outputs)

        # compute outputs, gradients, and update variables
        outputs = self.session.run(to_compute, feed_dict=inputs)

        for volume_type in volume_outputs:
            spec = self.spec[volume_type].copy()
            spec.roi = request[volume_type].roi
            batch.volumes[volume_type] = Volume(
                outputs[volume_type],
                spec)

        batch.loss = outputs['loss']
        batch.iteration = outputs['iteration'][0]

        if batch.iteration%self.save_every == 0:

            checkpoint_name = os.path.join(self.checkpoint_dir,
                self.meta_graph_filename +
                '_checkpoint_%i'%batch.iteration)

            logger.info(
                "Creating checkpoint %s",
                checkpoint_name)

            self.full_saver.save(
                self.session,
                checkpoint_name)

    def stop(self):

        if self.session is not None:

            self.optimizer = None
            self.loss = None

            self.session.close()
            self.graph = None
            self.session = None

    def __read_meta_graph(self):

        logger.info("Reading meta-graph...")

        # read the original meta-graph
        tf.train.import_meta_graph( self.meta_graph_filename + '.meta',
            clear_devices=True)

        # add custom gunpowder variables
        with tf.variable_scope('gunpowder'):
            self.iteration = tf.get_variable(
                'iteration',
                shape=1,
                initializer=tf.zeros_initializer,
                trainable=False)
            self.iteration_increment = tf.assign(
                self.iteration,
                self.iteration + 1)

        # Until now, only variables have been added to the graph that are part
        # of every checkpoint. We create a 'basic_saver' for only those
        # variables.
        self.basic_saver = tf.train.Saver(max_to_keep=None)

        # Add custom optimizer and loss, if requested. This potentially adds
        # more variables, not covered by the basic_saver.
        if self.optimizer_func is not None:
            loss, optimizer = self.optimizer_func(self.graph)
            self.loss = loss
            self.optimizer = optimizer

        # We create a 'full_saver' including those variables.
        self.full_saver = tf.train.Saver(max_to_keep=None)

        # find most recent checkpoint
        checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        logger.debug("Checkpoint Dir Contents %s", os.listdir(self.checkpoint_dir))

        if checkpoint:

            try:
                # Try to restore the graph, including the custom optimizer
                # state (if a custom optimizer was used).
                self.__restore_graph(checkpoint, restore_full=True)

            except tf.errors.NotFoundError:

                # If that failed, we just transitioned from an earlier training
                # without the custom optimizer. In this case, restore only the
                # variables of the original meta-graph and 'gunpowder'
                # variables. Custom optimizer variables will be default
                # initialized.
                logger.info("Checkpoint did not contain custom optimizer "
                            "variables")
                self.__restore_graph(checkpoint, restore_full=False)
        else:

            logger.info("No checkpoint found in %s", self.checkpoint_dir)

            # initialize all variables
            self.session.run(tf.global_variables_initializer())

    def __restore_graph(self, checkpoint, restore_full):

        logger.info("Restoring model from %s", checkpoint)

        if restore_full:

            logger.info("...using a saver for all variables")
            self.full_saver.restore(self.session, checkpoint)

        else:

            # initialize all variables, such that non-basic variables are
            # initialized
            self.session.run(tf.global_variables_initializer())

            logger.info("...using a saver for basic variables only")
            self.basic_saver.restore(self.session, checkpoint)

    def __collect_requested_outputs(self, request):

        volume_outputs = {}

        for output_name, volume_type in self.outputs.items():
            if volume_type in request:
                volume_outputs[volume_type] = output_name

        for output_name, volume_type in self.gradients.items():
            if volume_type in request:
                volume_outputs[volume_type] = self.tf_gradient[output_name]

        return volume_outputs

    def __collect_provided_inputs(self, batch):

        inputs = {}

        for input_name, input_type in self.inputs.items():
            if isinstance(input_type, VolumeType):
                if input_type in batch.volumes:
                    inputs[input_name] = batch.volumes[input_type].data
                else:
                    logger.warn("batch does not contain %s, input %s will not "
                                "be set", input_type, input_name)
            elif isinstance(input_type, np.ndarray):
                inputs[input_name] = input_type
            elif isinstance(input_type, str):
                inputs[input_name] = getattr(batch, input_type)
            else:
                raise Exception(
                    "Unknown network input type {}, can't be given to "
                    "network".format(input_type))

        return inputs
