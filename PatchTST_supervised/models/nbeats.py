"""
N-BEATS
-------
"""

from enum import Enum
from typing import List, NewType, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from layers.RevIN import RevIN

from einops import rearrange





ACTIVATIONS = [
    "ReLU",
    "RReLU",
    "PReLU",
    "ELU",
    "Softplus",
    "Tanh",
    "SELU",
    "LeakyReLU",
    "Sigmoid",
    "GELU",
]


class _GType(Enum):
    GENERIC = 1
    TREND = 2
    SEASONALITY = 3


GTypes = NewType("GTypes", _GType)


class _TrendGenerator(nn.Module):
    def __init__(self, expansion_coefficient_dim, target_length):
        super().__init__()

        # basis is of size (expansion_coefficient_dim, target_length)
        basis = torch.stack(
            [
                (torch.arange(target_length) / target_length) ** i
                for i in range(expansion_coefficient_dim)
            ],
            dim=1,
        ).T

        self.basis = nn.Parameter(basis, requires_grad=False)

    def forward(self, x):
        return torch.matmul(x, self.basis)


class _SeasonalityGenerator(nn.Module):
    def __init__(self, target_length):
        super().__init__()
        half_minus_one = int(target_length / 2 - 1)
        cos_vectors = [
            torch.cos(torch.arange(target_length) / target_length * 2 * np.pi * i)
            for i in range(1, half_minus_one + 1)
        ]
        sin_vectors = [
            torch.sin(torch.arange(target_length) / target_length * 2 * np.pi * i)
            for i in range(1, half_minus_one + 1)
        ]

        # basis is of size (2 * int(target_length / 2 - 1) + 1, target_length)
        basis = torch.stack(
            [torch.ones(target_length)] + cos_vectors + sin_vectors, dim=1
        ).T

        self.basis = nn.Parameter(basis, requires_grad=False)

    def forward(self, x):
        return torch.matmul(x, self.basis)


class _Block(nn.Module):
    def __init__(
        self,
        num_layers: int,
        layer_width: int,
        nr_params: int,
        expansion_coefficient_dim: int,
        input_chunk_length: int,
        target_length: int,
        g_type: GTypes,
        batch_norm: bool,
        dropout: float,
        activation: str,
    ):
        """PyTorch module implementing the basic building block of the N-BEATS architecture.
        The blocks produce outputs of size (target_length, nr_params); i.e.
        "one vector per parameter". The parameters are predicted only for forecast outputs.
        Backcast outputs are in the original "domain".
        Parameters
        ----------
        num_layers
            The number of fully connected layers preceding the final forking layers.
        layer_width
            The number of neurons that make up each fully connected layer.
        nr_params
            The number of parameters of the likelihood (or 1 if no likelihood is used)
        expansion_coefficient_dim
            The dimensionality of the waveform generator parameters, also known as expansion coefficients.
            Used in the generic architecture and the trend module of the interpretable architecture, where it determines
            the degree of the polynomial basis.
        input_chunk_length
            The length of the input sequence fed to the model.
        target_length
            The length of the forecast of the model.
        g_type
            The type of function that is implemented by the waveform generator.
        batch_norm
            Whether to use batch norm
        dropout
            Dropout probability
        activation
            The activation function of encoder/decoder intermediate layer.
        Inputs
        ------
        x of shape `(batch_size, input_chunk_length)`
            Tensor containing the input sequence.
        Outputs
        -------
        x_hat of shape `(batch_size, input_chunk_length)`
            Tensor containing the 'backcast' of the block, which represents an approximation of `x`
            given the constraints of the functional space determined by `g`.
        y_hat of shape `(batch_size, output_chunk_length)`
            Tensor containing the forward forecast of the block.
        """
        super().__init__()

        self.num_layers = num_layers
        self.layer_width = layer_width
        self.target_length = target_length
        self.nr_params = nr_params
        self.g_type = g_type
        self.dropout = dropout
        self.batch_norm = batch_norm

        
        self.activation = getattr(nn, activation)()

        # fully connected stack before fork
        self.linear_layer_stack_list = [nn.Linear(input_chunk_length, layer_width)]
        for _ in range(num_layers - 1):
            self.linear_layer_stack_list.append(nn.Linear(layer_width, layer_width))
            # self.linear_layer_stack_list.append(nn.TransformerEncoderLayer(d_model=layer_width, nhead=8))
            if self.batch_norm:
                self.linear_layer_stack_list.append(
                    nn.BatchNorm1d(num_features=self.layer_width)
                )

            if self.dropout > 0:
                self.linear_layer_stack_list.append(nn.Dropout(p=self.dropout))

        self.fc_stack = nn.ModuleList(self.linear_layer_stack_list)

        # Fully connected layer producing forecast/backcast expansion coefficients (waveform generator parameters).
        # The coefficients are emitted for each parameter of the likelihood.
        if g_type == _GType.SEASONALITY:
            self.backcast_linear_layer = nn.Linear(
                layer_width, 2 * int(input_chunk_length / 2 - 1) + 1
            )
            self.forecast_linear_layer = nn.Linear(
                layer_width, nr_params * (2 * int(target_length / 2 - 1) + 1)
            )
        else:
            self.backcast_linear_layer = nn.Linear(
                layer_width, expansion_coefficient_dim
            )
            self.forecast_linear_layer = nn.Linear(
                layer_width, nr_params * expansion_coefficient_dim
            )

        # waveform generator functions
        if g_type == _GType.GENERIC:
            self.backcast_g = nn.Linear(expansion_coefficient_dim, input_chunk_length)
            self.forecast_g = nn.Linear(expansion_coefficient_dim, target_length)
        elif g_type == _GType.TREND:
            self.backcast_g = _TrendGenerator(
                expansion_coefficient_dim, input_chunk_length
            )
            self.forecast_g = _TrendGenerator(expansion_coefficient_dim, target_length)
        elif g_type == _GType.SEASONALITY:
            self.backcast_g = _SeasonalityGenerator(input_chunk_length)
            self.forecast_g = _SeasonalityGenerator(target_length)
        

    def forward(self, x):
        batch_size = x.shape[0]

        # fully connected layer stack
        for layer in self.linear_layer_stack_list:
            x = self.activation(layer(x))

        # forked linear layers producing waveform generator parameters
        theta_backcast = self.backcast_linear_layer(x)
        theta_forecast = self.forecast_linear_layer(x)

        # set the expansion coefs in last dimension for the forecasts
        # theta_forecast = theta_forecast.view(batch_size, self.nr_params, -1)

        # waveform generator applications (project the expansion coefs onto basis vectors)
        x_hat = self.backcast_g(theta_backcast)
        y_hat = self.forecast_g(theta_forecast)

        # Set the distribution parameters as the last dimension
        # y_hat = y_hat.reshape(x.shape[0], self.target_length, self.nr_params)

        return x_hat, y_hat


class _Stack(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        num_layers: int,
        layer_width: int,
        nr_params: int,
        expansion_coefficient_dim: int,
        input_chunk_length: int,
        target_length: int,
        g_type: GTypes,
        batch_norm: bool,
        dropout: float,
        activation: str,
    ):
        """PyTorch module implementing one stack of the N-BEATS architecture that comprises multiple basic blocks.
        Parameters
        ----------
        num_blocks
            The number of blocks making up this stack.
        num_layers
            The number of fully connected layers preceding the final forking layers in each block.
        layer_width
            The number of neurons that make up each fully connected layer in each block.
        nr_params
            The number of parameters of the likelihood (or 1 if no likelihood is used)
        expansion_coefficient_dim
            The dimensionality of the waveform generator parameters, also known as expansion coefficients.
        input_chunk_length
            The length of the input sequence fed to the model.
        target_length
            The length of the forecast of the model.
        g_type
            The function that is implemented by the waveform generators in each block.
        batch_norm
            whether to apply batch norm on first block of this stack
        dropout
            Dropout probability
        activation
            The activation function of encoder/decoder intermediate layer.
        Inputs
        ------
        stack_input of shape `(batch_size, input_chunk_length)`
            Tensor containing the input sequence.
        Outputs
        -------
        stack_residual of shape `(batch_size, input_chunk_length)`
            Tensor containing the 'backcast' of the block, which represents an approximation of `x`
            given the constraints of the functional space determined by `g`.
        stack_forecast of shape `(batch_size, output_chunk_length)`
            Tensor containing the forward forecast of the stack.
        """
        super().__init__()

        self.input_chunk_length = input_chunk_length
        self.target_length = target_length
        self.nr_params = nr_params
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.activation = activation

        if g_type == _GType.GENERIC:
            
            self.blocks_list = _Block(
                    num_layers,
                    layer_width,
                    nr_params,
                    expansion_coefficient_dim,
                    input_chunk_length,
                    target_length,
                    g_type,
                    batch_norm=(
                        self.batch_norm and i == 0
                    ),  # batch norm only on first block of first stack
                    dropout=self.dropout,
                    activation=self.activation,
                )
               
        else:
            # same block instance is used for weight sharing
            interpretable_block = _Block(
                num_layers,
                layer_width,
                nr_params,
                expansion_coefficient_dim,
                input_chunk_length,
                target_length,
                g_type,
                batch_norm=self.batch_norm,
                dropout=self.dropout,
                activation=self.activation,
            )
            self.blocks_list = interpretable_block

        self.blocks = self.blocks_list
        self.num_blocks=num_blocks
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=16, nhead=1)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        self.transformer_encoder_2 = nn.TransformerEncoderLayer(d_model=336, nhead=8)
    def forward(self, x):
        # One forecast vector per parameter in the distribution
        stack_forecast = torch.zeros(
            x.shape[0],
            21,
            16,
            device=x.device,
            dtype=x.dtype,
        )
        # stack_forecast = torch.zeros(
        #     x.shape[0],
        #     256,
        #     37,
        #     device=x.device,
        #     dtype=x.dtype,
        # )

        for i in range(self.num_blocks):
            # pass input through block
            x_hat, y_hat = self.blocks(x)
            
            #Transformer layer
            y_hat=self.transformer_encoder(y_hat)
            # x_hat=self.transformer_encoder_2(x_hat)
            # add block forecast to stack forecast
            stack_forecast = stack_forecast + y_hat

            # subtract backcast from input to produce residual
            x = x - x_hat

        stack_residual = x

        return stack_residual, stack_forecast


class _NBEATSModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        nr_params: int,
        generic_architecture: bool,
        num_stacks: int,
        num_blocks: int,
        num_layers: int,
        layer_widths: List[int],
        expansion_coefficient_dim: int,
        trend_polynomial_degree: int,
        batch_norm: bool,
        dropout: float,
        activation: str,
        input_chunk_length:int,
        output_chunk_length:int
       
    ):
        """PyTorch module implementing the N-BEATS architecture.
        Parameters
        ----------
        output_dim
            Number of output components in the target
        nr_params
            The number of parameters of the likelihood (or 1 if no likelihood is used).
        generic_architecture
            Boolean value indicating whether the generic architecture of N-BEATS is used.
            If not, the interpretable architecture outlined in the paper (consisting of one trend
            and one seasonality stack with appropriate waveform generator functions).
        num_stacks
            The number of stacks that make up the whole model. Only used if `generic_architecture` is set to `True`.
        num_blocks
            The number of blocks making up every stack.
        num_layers
            The number of fully connected layers preceding the final forking layers in each block of every stack.
        layer_widths
            Determines the number of neurons that make up each fully connected layer in each block of every stack.
            If a list is passed, it must have a length equal to `num_stacks` and every entry in that list corresponds
            to the layer width of the corresponding stack. If an integer is passed, every stack will have blocks
            with FC layers of the same width.
        expansion_coefficient_dim
            The dimensionality of the waveform generator parameters, also known as expansion coefficients.
            Only used if `generic_architecture` is set to `True`.
        trend_polynomial_degree
            The degree of the polynomial used as waveform generator in trend stacks. Only used if
            `generic_architecture` is set to `False`.
        batch_norm
            Whether to apply batch norm on first block of the first stack
        dropout
            Dropout probability
        activation
            The activation function of encoder/decoder intermediate layer.
        **kwargs
            all parameters required for :class:`darts.model.forecasting_models.PLForecastingModule` base class.
        Inputs
        ------
        x of shape `(batch_size, input_chunk_length)`
            Tensor containing the input sequence.
        Outputs
        -------
        y of shape `(batch_size, output_chunk_length, target_size/output_dim, nr_params)`
            Tensor containing the output of the NBEATS module.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nr_params = nr_params
        self.output_chunk_length=output_chunk_length
        self.input_chunk_length=input_chunk_length
        self.input_chunk_length_multi = input_chunk_length 
        self.target_length = output_chunk_length 
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.activation = activation
        self.num_stacks =num_stacks
        if generic_architecture:
            self.stacks_list =_Stack(
                    num_blocks,
                    num_layers,
                    layer_widths[0],
                    nr_params,
                    expansion_coefficient_dim,
                    self.input_chunk_length_multi,
                    self.target_length,
                    _GType.GENERIC,
                    batch_norm=self.batch_norm, 
                      # batch norm only on first block of first stack
                    dropout=self.dropout,
                    activation=self.activation,
            )
                
            
        else:
            num_stacks = 2
            trend_stack = _Stack(
                num_blocks,
                num_layers,
                layer_widths[0],
                nr_params,
                trend_polynomial_degree + 1,
                self.input_chunk_length_multi,
                self.target_length,
                _GType.TREND,
                batch_norm=self.batch_norm,
                dropout=self.dropout,
                activation=self.activation,
            )
            seasonality_stack = _Stack(
                num_blocks,
                num_layers,
                layer_widths[1],
                nr_params,
                -1,
                self.input_chunk_length_multi,
                self.target_length,
                _GType.SEASONALITY,
                batch_norm=self.batch_norm,
                dropout=self.dropout,
                activation=self.activation,
            )
            self.stacks_list = [trend_stack, seasonality_stack]

        self.stacks = self.stacks_list
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=96, nhead=8)        
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        # setting the last backcast "branch" to be not trainable (without next block/stack, it doesn't need to be
        # backpropagated). Removing this lines would cause logtensorboard to crash, since no gradient is stored
        # on this params (the last block backcast is not part of the final output of the net).
        # self.stacks_list[-1].blocks[-1].backcast_linear_layer.requires_grad_(False)
        # self.stacks_list[-1].blocks[-1].backcast_g.requires_grad_(False)

    def forward(self, x_in):
        x = x_in

        # if x1, x2,... y1, y2... is one multivariate ts containing x and y, and a1, a2... one covariate ts
        # we reshape into x1, y1, a1, x2, y2, a2... etc
        batch=x.shape[0]
        # x = torch.reshape(x, (x.shape[0], self.input_chunk_length*self.input_dim, 1))
        # squeeze last dimension (because model is univariate)
        x = x.squeeze(dim=2)

        # One vector of length target_length per parameter in the distribution
        y = torch.zeros(
            x.shape[0],
            21,
            16,
            device=x.device,
            dtype=x.dtype,
        )

        # y = torch.zeros(
        #     x.shape[0],
        #     256,
        #     37,
        #     device=x.device,
        #     dtype=x.dtype,
        # )

        for i in range(self.num_stacks):
            # compute stack output
            stack_residual, stack_forecast = self.stacks(x)

            # stack_forecast=self.transformer_encoder(stack_forecast)
            # add stack forecast to final output
            y = y + stack_forecast

            # set current stack residual as input for next stack
            x = stack_residual

        # In multivariate case, we get a result [x1_param1, x1_param2], [y1_param1, y1_param2], [x2..], [y2..], ...
        # We want to reshape to original format. We also get rid of the covariates and keep only the target dimensions.
        # The covariates are by construction added as extra time series on the right side. So we need to get rid of this
        # right output (keeping only :self.output_dim).
        # y = y.view(
        #     batch, self.output_chunk_length, self.output_dim
        # )[:, :, : self.output_dim]

        return y
    

class Model(nn.Module):
    def __init__(
        self,
        args,
        
        generic_architecture: bool = True,
        # num_stacks: int = 30,
        num_stacks: int = 3,
        num_blocks: int = 1,
        num_layers: int = 2,
        # num_layers: int = 8,
        layer_widths: Union[int, List[int]] = 256,
        expansion_coefficient_dim: int = 5,
        trend_polynomial_degree: int = 2,
        dropout: float = 0.0,
        # dropout: float = 0.1,
        activation: str = "GELU",
        **kwargs,
    ):
        """Neural Basis Expansion Analysis Time Series Forecasting (N-BEATS).
        This is an implementation of the N-BEATS architecture, as outlined in [1]_.
        In addition to the univariate version presented in the paper, our implementation also
        supports multivariate series (and covariates) by flattening the model inputs to a 1-D series
        and reshaping the outputs to a tensor of appropriate dimensions. Furthermore, it also
        supports producing probabilistic forecasts (by specifying a `likelihood` parameter).
        This model supports past covariates (known for `input_chunk_length` points before prediction time).
        Parameters
        ----------
        input_chunk_length
            The length of the input sequence fed to the model.
        output_chunk_length
            The length of the forecast of the model.
        generic_architecture
            Boolean value indicating whether the generic architecture of N-BEATS is used.
            If not, the interpretable architecture outlined in the paper (consisting of one trend
            and one seasonality stack with appropriate waveform generator functions).
        num_stacks
            The number of stacks that make up the whole model. Only used if `generic_architecture` is set to `True`.
            The interpretable architecture always uses two stacks - one for trend and one for seasonality.
        num_blocks
            The number of blocks making up every stack.
        num_layers
            The number of fully connected layers preceding the final forking layers in each block of every stack.
        layer_widths
            Determines the number of neurons that make up each fully connected layer in each block of every stack.
            If a list is passed, it must have a length equal to `num_stacks` and every entry in that list corresponds
            to the layer width of the corresponding stack. If an integer is passed, every stack will have blocks
            with FC layers of the same width.
        expansion_coefficient_dim
            The dimensionality of the waveform generator parameters, also known as expansion coefficients.
            Only used if `generic_architecture` is set to `True`.
        trend_polynomial_degree
            The degree of the polynomial used as waveform generator in trend stacks. Only used if
            `generic_architecture` is set to `False`.
        dropout
            The dropout probability to be used in fully connected layers. This is compatible with Monte Carlo dropout
            at inference time for model uncertainty estimation (enabled with ``mc_dropout=True`` at
            prediction time).
        activation
            The activation function of encoder/decoder intermediate layer (default='ReLU').
            Supported activations: ['ReLU','RReLU', 'PReLU', 'Softplus', 'Tanh', 'SELU', 'LeakyReLU',  'Sigmoid']
        **kwargs
            Optional arguments to initialize the pytorch_lightning.Module, pytorch_lightning.Trainer, and
            Darts' :class:`TorchForecastingModel`.
        loss_fn
            PyTorch loss function used for training.
            This parameter will be ignored for probabilistic models if the ``likelihood`` parameter is specified.
            Default: ``torch.nn.MSELoss()``.
        likelihood
            One of Darts' :meth:`Likelihood <darts.utils.likelihood_models.Likelihood>` models to be used for
            probabilistic forecasts. Default: ``None``.
        torch_metrics
            A torch metric or a ``MetricCollection`` used for evaluation. A full list of available metrics can be found
            at https://torchmetrics.readthedocs.io/en/latest/. Default: ``None``.
        optimizer_cls
            The PyTorch optimizer class to be used. Default: ``torch.optim.Adam``.
        optimizer_kwargs
            Optionally, some keyword arguments for the PyTorch optimizer (e.g., ``{'lr': 1e-3}``
            for specifying a learning rate). Otherwise the default values of the selected ``optimizer_cls``
            will be used. Default: ``None``.
        lr_scheduler_cls
            Optionally, the PyTorch learning rate scheduler class to be used. Specifying ``None`` corresponds
            to using a constant learning rate. Default: ``None``.
        lr_scheduler_kwargs
            Optionally, some keyword arguments for the PyTorch learning rate scheduler. Default: ``None``.
        batch_size
            Number of time series (input and output sequences) used in each training pass. Default: ``32``.
        n_epochs
            Number of epochs over which to train the model. Default: ``100``.
        model_name
            Name of the model. Used for creating checkpoints and saving tensorboard data. If not specified,
            defaults to the following string ``"YYYY-mm-dd_HH_MM_SS_torch_model_run_PID"``, where the initial part
            of the name is formatted with the local date and time, while PID is the processed ID (preventing models
            spawned at the same time by different processes to share the same model_name). E.g.,
            ``"2021-06-14_09_53_32_torch_model_run_44607"``.
        work_dir
            Path of the working directory, where to save checkpoints and Tensorboard summaries.
            Default: current working directory.
        log_tensorboard
            If set, use Tensorboard to log the different parameters. The logs will be located in:
            ``"{work_dir}/darts_logs/{model_name}/logs/"``. Default: ``False``.
        nr_epochs_val_period
            Number of epochs to wait before evaluating the validation loss (if a validation
            ``TimeSeries`` is passed to the :func:`fit()` method). Default: ``1``.
        force_reset
            If set to ``True``, any previously-existing model with the same name will be reset (all checkpoints will
            be discarded). Default: ``False``.
        save_checkpoints
            Whether or not to automatically save the untrained model and checkpoints from training.
            To load the model from checkpoint, call :func:`MyModelClass.load_from_checkpoint()`, where
            :class:`MyModelClass` is the :class:`TorchForecastingModel` class that was used (such as :class:`TFTModel`,
            :class:`NBEATSModel`, etc.). If set to ``False``, the model can still be manually saved using
            :func:`save()` and loaded using :func:`load()`. Default: ``False``.
        add_encoders
            A large number of past and future covariates can be automatically generated with `add_encoders`.
            This can be done by adding multiple pre-defined index encoders and/or custom user-made functions that
            will be used as index encoders. Additionally, a transformer such as Darts' :class:`Scaler` can be added to
            transform the generated covariates. This happens all under one hood and only needs to be specified at
            model creation.
            Read :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>` to find out more about
            ``add_encoders``. Default: ``None``. An example showing some of ``add_encoders`` features:
            .. highlight:: python
            .. code-block:: python
                add_encoders={
                    'cyclic': {'future': ['month']},
                    'datetime_attribute': {'future': ['hour', 'dayofweek']},
                    'position': {'past': ['relative'], 'future': ['relative']},
                    'custom': {'past': [lambda idx: (idx.year - 1950) / 50]},
                    'transformer': Scaler()
                }
            ..
        random_state
            Control the randomness of the weights initialization. Check this
            `link <https://scikit-learn.org/stable/glossary.html#term-random_state>`_ for more details.
            Default: ``None``.
        pl_trainer_kwargs
            By default :class:`TorchForecastingModel` creates a PyTorch Lightning Trainer with several useful presets
            that performs the training, validation and prediction processes. These presets include automatic
            checkpointing, tensorboard logging, setting the torch device and more.
            With ``pl_trainer_kwargs`` you can add additional kwargs to instantiate the PyTorch Lightning trainer
            object. Check the `PL Trainer documentation
            <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_ for more information about the
            supported kwargs. Default: ``None``.
            Running on GPU(s) is also possible using ``pl_trainer_kwargs`` by specifying keys ``"accelerator",
            "devices", and "auto_select_gpus"``. Some examples for setting the devices inside the ``pl_trainer_kwargs``
            dict:
            - ``{"accelerator": "cpu"}`` for CPU,
            - ``{"accelerator": "gpu", "devices": [i]}`` to use only GPU ``i`` (``i`` must be an integer),
            - ``{"accelerator": "gpu", "devices": -1, "auto_select_gpus": True}`` to use all available GPUS.
            For more info, see here:
            https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags , and
            https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_basic.html#train-on-multiple-gpus
            With parameter ``"callbacks"`` you can add custom or PyTorch-Lightning built-in callbacks to Darts'
            :class:`TorchForecastingModel`. Below is an example for adding EarlyStopping to the training process.
            The model will stop training early if the validation loss `val_loss` does not improve beyond
            specifications. For more information on callbacks, visit:
            `PyTorch Lightning Callbacks
            <https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html>`_
            .. highlight:: python
            .. code-block:: python
                from pytorch_lightning.callbacks.early_stopping import EarlyStopping
                # stop training when validation loss does not decrease more than 0.05 (`min_delta`) over
                # a period of 5 epochs (`patience`)
                my_stopper = EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    min_delta=0.05,
                    mode='min',
                )
                pl_trainer_kwargs={"callbacks": [my_stopper]}
            ..
            Note that you can also use a custom PyTorch Lightning Trainer for training and prediction with optional
            parameter ``trainer`` in :func:`fit()` and :func:`predict()`.
        show_warnings
            whether to show warnings raised from PyTorch Lightning. Useful to detect potential issues of
            your forecasting use case. Default: ``False``.
        References
        ----------
        .. [1] https://openreview.net/forum?id=r1ecqn4YwB
        """
        super().__init__()

        # extract pytorch lightning module kwargs
        

        # input_chunk_length=args.seq_len
        # output_chunk_length=args.pred_len
        self.patch_len=16
        self.stride=16
        input_chunk_length=self.patch_len*
        output_chunk_length=self.patch_len
        patch_num = int(( args.seq_len- self.patch_len)/self.stride + 1)
        self.input_dim=patch_num
        self.output_dim=patch_num

        self.generic_architecture = generic_architecture
        self.num_stacks = num_stacks
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.layer_widths = layer_widths
        self.expansion_coefficient_dim = expansion_coefficient_dim
        self.trend_polynomial_degree = trend_polynomial_degree

        # Currently batch norm is not an option as it seems to perform badly
        self.batch_norm = False

        self.dropout = dropout
        self.activation = activation

        if not generic_architecture:
            self.num_stacks = 2

        if isinstance(layer_widths, int):
            self.layer_widths = [layer_widths] * self.num_stacks

        # self.input_dim=args.seq_len
        # self.output_dim=args.pred_len
        # self.input_dim=args.pred_len
        # self.output_dim=args.pred_len

        self.nr_params = 1
        self.model=_NBEATSModule(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            nr_params=self.nr_params,
            generic_architecture=self.generic_architecture,
            num_stacks=self.num_stacks,
            num_blocks=self.num_blocks,
            num_layers=self.num_layers,
            layer_widths=self.layer_widths,
            expansion_coefficient_dim=self.expansion_coefficient_dim,
            trend_polynomial_degree=self.trend_polynomial_degree,
            batch_norm=self.batch_norm,
            dropout=self.dropout,
            activation=self.activation,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length
        )
        self.revin=RevIN(args.enc_in)
        self.out=nn.Linear(336,96)
        
    def forward(self,x,batch_x_mark=0, dec_inp=0, batch_y_mark=0, batch_y=0) :
        # samples are made of (past_target, past_covariates, future_target)
        batch=x.shape[0]
        x=self.revin(x,'norm')
        x=x.permute(0,2,1)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)   
        x=rearrange(x,'b d t l->b t (l d)')

        y=self.model(x)
        # y=rearrange(y,'(b d) t l-> b d (t l)',b=batch)
        # y=self.out(y)
        y=y.permute(0,2,1)
        y=self.revin(y,'denorm')
        # y=self.out(y.permute(0,2,1)).permute(0,2,1)
        return y

       
