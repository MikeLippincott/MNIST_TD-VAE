"""
Original code from Xinqiang Ding <xqding@umich.edu>"

Code makes model for TD-VAE
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class DBlock(nn.Module):
    """A basie building block for parametrize a normal distribution.
    It is corresponding to the D operation in the reference Appendix.

    Attributes
    ----------
    input_size : int
        The size of the input tensor.
    hidden_size : int
        The size of the hidden tensor.

    Methods
    -------
    forward(input) -> Tuple[torch.Tensor, torch.Tensor]
        Forward pass of the DBlock
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        """
        Initialize the DBlock

        Parameters
        ----------
        input_size : int
            The size of the input tensor.
        hidden_size : int
            The size of the hidden tensor.
        output_size :
            The size of the output tensor

        Returns
        -------
        None
        """
        super(DBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, output_size)
        self.fc_logsigma = nn.Linear(hidden_size, output_size)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the DBlock

        Parameters
        ----------
        input : torch.Tensor
            Input tensor to the DBlock

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple containing the mean and the log of the standard deviation of the input
        """
        t = torch.tanh(self.fc1(input))
        t = t * torch.sigmoid(self.fc2(input))
        mu = self.fc_mu(t)
        logsigma = self.fc_logsigma(t)
        return mu, logsigma


class PreProcess(nn.Module):
    """
    The pre-process layer for MNIST image

    Attributes
    ----------
    input_size : int
        The size of the input tensor.
    processed_x_size : int
        The size of the output tensor

    Methods
    -------
    forward(input)
        Forward pass of the pre-process layer

    """

    def __init__(self, input_size: int, processed_x_size: int) -> None:
        """
        Initialize the PreProcess layer

        Parameters
        ----------
        input_size : int
            The size of the input tensor.
        processed_x_size : int
            The size of the output tensor
        """
        super(PreProcess, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, processed_x_size)
        self.fc2 = nn.Linear(processed_x_size, processed_x_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the pre-process layer

        Parameters
        ----------
        input : torch.Tensor
            Input tensor to the pre-process layer

        Returns
        -------
        torch.Tensor
            Output tensor of the pre-process layer
        """
        t = torch.relu(self.fc1(input))
        t = torch.relu(self.fc2(t))
        return t


# Code adapted from: https://github.com/ankitkv/pylego/blob/master/pylego/ops.py
# because is cannot be installed via pip due to software decay
# copy and paste the code here
class MultilayerLSTMCell(nn.Module):
    """Provides a mutli-layer wrapper for LSTMCell.

    Attributes
    ----------
    hidden_size : int
        The size of the hidden tensor.
    layers : int
        Number of layers.
    every_layer_input : bool
        Consider raw input at every layer.
    use_previous_higher : bool
        Take higher layer at previous timestep as input to current layer.

    Methods
    -------
    forward(input_, hx=None) -> list
        Forward pass of the MultilayerLSTMCell

    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        layers: int = 1,
        every_layer_input: bool = False,
        use_previous_higher: bool = False,
    ) -> None:
        """
        Initialize the MultilayerLSTMCell

        Parameters
        ----------
        input_size : int
            The size of the input tensor.
        hidden_size : int
            The size of the hidden tensor.
        bias : bool, optional
            Bias for the LSTMCell, by default True
        layers : int, optional
            Number of layers, by default 1
        every_layer_input : bool, optional
            Consider raw input at every layer, by default False
        use_previous_higher : bool, optional
            Take higher layer at previous timestep as input to current layer, by default False

        Returns
        -------
        None
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = layers
        self.every_layer_input = every_layer_input
        self.use_previous_higher = use_previous_higher
        input_sizes = [input_size] + [hidden_size for _ in range(1, layers)]
        if every_layer_input:
            for i in range(1, layers):
                input_sizes[i] += input_size
        if use_previous_higher:
            for i in range(layers - 1):
                input_sizes[i] += hidden_size
        self.lstm_cells = nn.ModuleList(
            [nn.LSTMCell(input_sizes[i], hidden_size, bias=bias) for i in range(layers)]
        )

    def forward(self, input_: torch.tensor, hx: list = None) -> list:
        """
        Forward pass of the MultilayerLSTMCell

        Parameters
        ----------
        input_ : torch.tensor
            Input: input, [(h_0, c_0), ..., (h_L, c_L)]
        hx : list, optional
            Hidden state, by default None

        Returns
        -------
        list
            Output: [(h_0, c_0), ..., (h_L, c_L)]
        """
        if hx is None:
            hx = [None] * self.layers
        outputs = []
        recent = input_
        for layer in range(self.layers):
            if layer > 0 and self.every_layer_input:
                recent = torch.cat([recent, input_], dim=1)
            if layer < self.layers - 1 and self.use_previous_higher:
                if hx[layer + 1] is None:
                    prev = recent.new_zeros([recent.size(0), self.hidden_size])
                else:
                    prev = hx[layer + 1][0]
                recent = torch.cat([recent, prev], dim=1)
            out = self.lstm_cells[layer](recent, hx[layer])
            recent = out[0]
            outputs.append(out)
        return outputs


# same here:
# Code adapted from:
# https://github.com/ankitkv/pylego/blob/master/pylego/ops.py
class MultilayerLSTM(nn.Module):
    """A multilayer LSTM that uses MultilayerLSTMCell.

    Attributes
    ----------
    input_size : int
        The size of the input tensor.
    hidden_size : int
        The size of the hidden tensor.
    layers : int
        Number of layers.
    every_layer_input : bool
        Consider raw input at every layer.
    use_previous_higher : bool
        Take higher layer at previous timestep as input to current layer.

    Methods
    -------
    forward(input_, reset=None) -> torch.tensor
        Forward pass of the Multilayer
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        layers: int = 1,
        every_layer_input: bool = False,
        use_previous_higher: bool = False,
    ) -> None:
        """
        Initialize the MultilayerLSTM

        Parameters
        ----------
        input_size : int
            The size of the input tensor.
        hidden_size : int
            The size of the hidden tensor.
        bias : bool, optional
            Set the bias as true, by default True
        layers : int, optional
            The number of layer for the model to have , by default 1
        every_layer_input : bool, optional
            Each layer has input from the previous layer, by default False
        use_previous_higher : bool, optional
            Use the previous higher layer as input, by default False

        Returns
        -------
        None

        """
        super().__init__()
        self.cell = MultilayerLSTMCell(
            input_size,
            hidden_size,
            bias=bias,
            layers=layers,
            every_layer_input=every_layer_input,
            use_previous_higher=use_previous_higher,
        )

    def forward(self, input_: torch.tensor, reset: torch.tensor = None) -> torch.tensor:
        """If reset is 1.0, the RNN state is reset AFTER that timestep's output is produced, otherwise if reset is 0.0,
        nothing is changed.

        Parameters
        ----------
        input_ : torch.tensor
            Input tensor to the MultilayerLSTM
        reset : torch.tensor, optional
            Reset the RNN state, by default None

        Returns
        -------
        torch.tensor
            Output tensor of the MultilayerLSTM
        """
        hx = None
        outputs = []
        for t in range(input_.size(1)):
            hx = self.cell(input_[:, t], hx)
            outputs.append(torch.cat([h[:, None, None, :] for (h, c) in hx], dim=2))
            if reset is not None:
                reset_t = reset[:, t, None]
                if torch.any(reset_t > 1e-6):
                    for i, (h, c) in enumerate(hx):
                        hx[i] = (h * (1.0 - reset_t), c * (1.0 - reset_t))

        return torch.cat(
            outputs, dim=1
        )  # size: batch_size, length, layers, hidden_size


# same here:
# Code adapted from:
# https://github.com/ankitkv/pylego/blob/master/pylego/ops.py
class GaussianTools:
    """
    A class for Gaussian operations.

    Methods
    -------
    reparameterize_gaussian(mu, logvar, sample=True, return_eps=False) -> Tuple[torch.tensor, torch.tensor] | torch.tensor
        Reparameterize a Gaussian distribution
    kl_div_gaussian(q_mu, q_logvar, p_mu=None, p_logvar=None) -> torch.tensor
        Batched KL divergence D(q||p) computation
    gaussian_log_prob(mu, logvar, x) -> torch.tensor
        Batched log probability log p(x) computation
    """

    def __init__(self):
        """
        Initialize the GaussianTools
        """
        self.LOG2PI = np.log(2 * np.pi)

    def reparameterize_gaussian(
        self,
        mu: torch.tensor,
        logvar: torch.tensor,
        sample: bool = True,
        return_eps: bool = False,
    ) -> Tuple[torch.tensor, torch.tensor] | torch.tensor:
        """
        Reparameterize a Gaussian distribution.

        Parameters
        ----------
        mu : torch.tensor
            Mean of the distribution
        logvar : torch.tensor
            Log of the variance of the distribution
        sample : bool, optional
            If true, then sampling is a go, by default True
        return_eps : bool, optional
            If true return epsilon, by default False

        Returns
        -------
        Tuple[torch.tensor, torch.tensor] | torch.tensor
            The reparameterized Gaussian distribution
        """
        std = torch.exp(0.5 * logvar)
        if sample:
            eps = torch.randn_like(std)
        else:
            eps = torch.zeros_like(std)
        ret = eps.mul(std).add_(mu)
        if return_eps:
            return ret, eps
        else:
            return ret

    def kl_div_gaussian(
        self,
        q_mu: torch.tensor,
        q_logvar: torch.tensor,
        p_mu: torch.tensor = None,
        p_logvar: torch.tensor = None,
    ) -> torch.tensor:
        """Batched KL divergence D(q||p) computation.

        Parameters
        ----------
        q_mu : torch.tensor
            Mean of the distribution q
        q_logvar : torch.tensor
            Log of the variance of the distribution q
        p_mu : torch.tensor, optional
            Mean of the distribution p, by default None
        p_logvar : torch.tensor, optional
            Log of the variance of the distribution p, by default None

        Returns
        -------
        torch.tensor
            KL divergence of the two distributions
        """
        if p_mu is None or p_logvar is None:
            zero = q_mu.new_zeros(1)
            p_mu = p_mu or zero
            p_logvar = p_logvar or zero
        logvar_diff = q_logvar - p_logvar
        kl_div = -0.5 * (
            1.0
            + logvar_diff
            - logvar_diff.exp()
            - ((q_mu - p_mu) ** 2 / p_logvar.exp())
        )
        return kl_div.sum(dim=-1)

    def gaussian_log_prob(
        self, mu: torch.tensor, logvar: torch.tensor, x: torch.tensor
    ) -> torch.tensor:
        """Batched log probability log p(x) computation.

        Parameters
        ----------
        mu : torch.tensor
            Mean of the distribution
        logvar : torch.tensor
            Log of the variance of the distribution
        x : torch.tensor
            Input tensor

        Returns
        -------
        torch.tensor
            Log probability of the distribution
        """
        logprob = -0.5 * (self.LOG2PI + logvar + ((x - mu) ** 2 / logvar.exp()))
        return logprob.sum(dim=-1)


class Decoder(nn.Module):
    """The decoder layer converting state to observation.
    Because the observation is MNIST image whose elements are values
    between 0 and 1, the output of this layer are probabilities of
    elements being 1.

    Attributes
    ----------
    z_size : int
        The size of the input tensor.
    hidden_size : int
        The size of the hidden tensor.
    x_size : int
        The size of the output tensor

    Methods
    -------
    forward(z)
        Forward pass of the decoder layer

    """

    def __init__(self, z_size: int, hidden_size: int, x_size: int) -> None:
        """
        Initialize the Decoder layer

        Parameters
        ----------
        z_size : int
            The size of the input tensor.
        hidden_size : int
            The size of the hidden tensor
        x_size : int
            The size of the output tensor
        """
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(z_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, x_size)

    def forward(self, z: torch.tensor) -> torch.tensor:
        """
        Forward pass of the decoder layer

        Parameters
        ----------
        z : torch.tensor
            The input tensor of the latent state

        Returns
        -------
        torch.tensor
            The output tensor of the decoder layer
        """
        t = torch.tanh(self.fc1(z))
        t = torch.tanh(self.fc2(t))
        p = torch.sigmoid(self.fc3(t))
        return p


class TD_VAE(nn.Module, GaussianTools):
    """The full TD_VAE model with jumpy prediction.

    First, let's first go through some definitions which would help
    understanding what is going on in the following code.

    Belief: As the model is feed a sequence of observations, x_t, the
      model updates its belief state, b_t, through a LSTM network. It
      is a deterministic function of x_t. We call b_t the belief at
      time t instead of belief state, becuase we call the hidden state z
      state.

    State: The latent state variable, z.

    Observation: The observated variable, x. In this case, it represents
      binarized MNIST images

    Attributes
    ----------
    x_size : int
        The size of the input tensor.
    processed_x_size : int
        The size of the processed input tensor.
    b_size : int
        The size of the belief tensor.
    z_size : int
        The size of the state tensor.
    d_block_hidden_size : int
        The size of the hidden tensor in DBlock
    decoder_hidden_size : int
        The size of the hidden tensor in Decoder

    Methods
    -------
    forward(images)
        Forward pass of the model
    calculate_loss(t1, t2)
        Calculate the loss of the model
    rollout(images, t1, t2)
        Rollout the model

    """

    def __init__(
        self,
        x_size: int,
        processed_x_size: int,
        b_size: int,
        z_size: int,
        d_block_hidden_size: int = 50,
        decoder_hidden_size: int = 200,
        layers: int = 2,
        samples_per_seq: int = 1,
        t_diff_min: int = 1,
        t_diff_max: int = 20,
    ):
        """ "
        Initialize the model with the given parameters

        Parameters
        ----------
        x_size : int
            The size of the input tensor.
        processed_x_size : int
            The size of the processed input tensor.
        b_size : int
            The size of the belief tensor.
        z_size : int
            The size of the state tensor.
        d_block_hidden_size : int, optional
            The size of the hidden tensor in DBlock, by default 50
        decoder_hidden_size : int, optional
            The size of the hidden tensor in Decoder, by default 200
        layers : int, optional
            The number of layers, by default 1
        samples_per_seq : int, optional
            The number of samples per sequence, by default 1
        t_diff_min : int, optional
            The minimum time difference, by default 1
        t_diff_max : int, optional
            The maximum time difference, by default 20
        """
        # Initialize the model with the given parameters
        super(TD_VAE, self).__init__()
        # init GaussianTools
        GaussianTools.__init__(self)

        self.x_size = x_size
        self.processed_x_size = processed_x_size
        self.b_size = b_size
        self.z_size = z_size
        self.d_block_hidden_size = d_block_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.layers = layers
        self.samples_per_seq = samples_per_seq
        self.t_diff_min = t_diff_min
        self.t_diff_max = t_diff_max

        ## Code adapted from: https://github.com/MaxASchwarzer/TDVAE-RL-Project/blob/master/models/tdvae/tdvae.py
        # Input pre-process layer
        self.process_x = PreProcess(x_size, processed_x_size)

        # Multilayer LSTM for aggregating belief states
        self.b_rnn = MultilayerLSTM(
            input_size=processed_x_size,
            hidden_size=b_size,
            layers=layers,
            every_layer_input=True,
            use_previous_higher=True,
        )

        # Multilayer state model is used. Sampling is done by sampling higher layers first.
        self.z_b = nn.ModuleList(
            [
                DBlock(
                    b_size + (z_size if layer < layers - 1 else 0),
                    d_block_hidden_size,
                    z_size,
                )
                for layer in range(layers)
            ]
        )

        # Given belief and state at time t2, infer the state at time t1
        self.z1_z2_b1 = nn.ModuleList(
            [
                DBlock(
                    b_size + layers * z_size + (z_size if layer < layers - 1 else 0),
                    d_block_hidden_size,
                    z_size,
                )
                for layer in range(layers)
            ]
        )

        # Given the state at time t1, model state at time t2 through state transition
        self.z2_z1 = nn.ModuleList(
            [
                DBlock(
                    layers * z_size + (z_size if layer < layers - 1 else 0),
                    d_block_hidden_size,
                    z_size,
                )
                for layer in range(layers)
            ]
        )

        # state to observation
        self.x_z = Decoder(layers * z_size, decoder_hidden_size, x_size)

    def forward(self, images: torch.Tensor) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Forward pass of the model

        Parameters
        ----------
        images : torch.Tensor
            The input tensor

        Returns
        -------
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor
        ]
            The output tensor of the model Tuple[


                images,
                t2,
                qs_z1_z2_b1_mu,
                qs_z1_z2_b1_logvar,
                pb_z1_b1_mu,
                pb_z1_b1_logvar,
                qb_z2_b2_mu,
                qb_z2_b2_logvar,
                qb_z2_b2,
                pt_z2_z1_mu,
                pt_z2_z1_logvar,
                pd_x2_z2,
                ]
        )
        """
        # sample t1 and t2
        t1 = torch.randint(
            0,
            images.size(1) - self.t_diff_max,
            (self.samples_per_seq, images.size(0)),
            device=images.device,
        )
        t2 = t1 + torch.randint(
            self.t_diff_min,
            self.t_diff_max + 1,
            (self.samples_per_seq, images.size(0)),
            device=images.device,
        )
        # images = images[:, :t2.max() + 1]  # usually not required with big enough batch size

        # pre-process image x
        processed_x = self.process_x(images)  # max x length is max(t2) + 1

        # aggregate the belief b
        b = self.b_rnn(processed_x)  # size: bs, time, layers, dim

        # replicate b multiple times
        b = b[None, ...].expand(
            self.samples_per_seq, -1, -1, -1, -1
        )  # size: copy, bs, time, layers, dim

        # Element-wise indexing. sizes: bs, layers, dim
        b1 = torch.gather(
            b, 2, t1[..., None, None, None].expand(-1, -1, -1, b.size(3), b.size(4))
        ).view(-1, b.size(3), b.size(4))
        b2 = torch.gather(
            b, 2, t2[..., None, None, None].expand(-1, -1, -1, b.size(3), b.size(4))
        ).view(-1, b.size(3), b.size(4))

        # q_B(z2 | b2)
        qb_z2_b2_mus, qb_z2_b2_logvars, qb_z2_b2s = [], [], []
        for layer in range(self.layers - 1, -1, -1):
            if layer == self.layers - 1:
                qb_z2_b2_mu, qb_z2_b2_logvar = self.z_b[layer](b2[:, layer])
            else:
                qb_z2_b2_mu, qb_z2_b2_logvar = self.z_b[layer](
                    torch.cat([b2[:, layer], qb_z2_b2], dim=1)
                )
            qb_z2_b2_mus.insert(0, qb_z2_b2_mu)
            qb_z2_b2_logvars.insert(0, qb_z2_b2_logvar)

            qb_z2_b2 = self.reparameterize_gaussian(
                qb_z2_b2_mu, qb_z2_b2_logvar, self.training
            )
            qb_z2_b2s.insert(0, qb_z2_b2)

        qb_z2_b2_mu = torch.cat(qb_z2_b2_mus, dim=1)
        qb_z2_b2_logvar = torch.cat(qb_z2_b2_logvars, dim=1)
        qb_z2_b2 = torch.cat(qb_z2_b2s, dim=1)

        # q_S(z1 | z2, b1, b2) ~= q_S(z1 | z2, b1)
        qs_z1_z2_b1_mus, qs_z1_z2_b1_logvars, qs_z1_z2_b1s = [], [], []
        for layer in range(
            self.layers - 1, -1, -1
        ):  # TODO optionally condition t2 - t1
            if layer == self.layers - 1:
                qs_z1_z2_b1_mu, qs_z1_z2_b1_logvar = self.z1_z2_b1[layer](
                    torch.cat([qb_z2_b2, b1[:, layer]], dim=1)
                )
            else:
                qs_z1_z2_b1_mu, qs_z1_z2_b1_logvar = self.z1_z2_b1[layer](
                    torch.cat([qb_z2_b2, b1[:, layer], qs_z1_z2_b1], dim=1)
                )
            qs_z1_z2_b1_mus.insert(0, qs_z1_z2_b1_mu)
            qs_z1_z2_b1_logvars.insert(0, qs_z1_z2_b1_logvar)

            qs_z1_z2_b1 = self.reparameterize_gaussian(
                qs_z1_z2_b1_mu, qs_z1_z2_b1_logvar, self.training
            )
            qs_z1_z2_b1s.insert(0, qs_z1_z2_b1)

        qs_z1_z2_b1_mu = torch.cat(qs_z1_z2_b1_mus, dim=1)
        qs_z1_z2_b1_logvar = torch.cat(qs_z1_z2_b1_logvars, dim=1)
        qs_z1_z2_b1 = torch.cat(qs_z1_z2_b1s, dim=1)

        # p_T(z2 | z1), also conditions on q_B(z2) from higher layer
        pt_z2_z1_mus, pt_z2_z1_logvars = [], []
        for layer in range(
            self.layers - 1, -1, -1
        ):  # TODO optionally condition t2 - t1
            if layer == self.layers - 1:
                pt_z2_z1_mu, pt_z2_z1_logvar = self.z2_z1[layer](qs_z1_z2_b1)
            else:
                pt_z2_z1_mu, pt_z2_z1_logvar = self.z2_z1[layer](
                    torch.cat([qs_z1_z2_b1, qb_z2_b2s[layer + 1]], dim=1)
                )
            pt_z2_z1_mus.insert(0, pt_z2_z1_mu)
            pt_z2_z1_logvars.insert(0, pt_z2_z1_logvar)

        pt_z2_z1_mu = torch.cat(pt_z2_z1_mus, dim=1)
        pt_z2_z1_logvar = torch.cat(pt_z2_z1_logvars, dim=1)

        # p_B(z1 | b1)
        pb_z1_b1_mus, pb_z1_b1_logvars = [], []
        for layer in range(
            self.layers - 1, -1, -1
        ):  # TODO optionally condition t2 - t1
            if layer == self.layers - 1:
                pb_z1_b1_mu, pb_z1_b1_logvar = self.z_b[layer](b1[:, layer])
            else:
                pb_z1_b1_mu, pb_z1_b1_logvar = self.z_b[layer](
                    torch.cat([b1[:, layer], qs_z1_z2_b1s[layer + 1]], dim=1)
                )
            pb_z1_b1_mus.insert(0, pb_z1_b1_mu)
            pb_z1_b1_logvars.insert(0, pb_z1_b1_logvar)

        pb_z1_b1_mu = torch.cat(pb_z1_b1_mus, dim=1)
        pb_z1_b1_logvar = torch.cat(pb_z1_b1_logvars, dim=1)

        # p_D(x2 | z2)
        pd_x2_z2 = self.x_z(qb_z2_b2)

        return (
            images,
            t2,
            qs_z1_z2_b1_mu,
            qs_z1_z2_b1_logvar,
            pb_z1_b1_mu,
            pb_z1_b1_logvar,
            qb_z2_b2_mu,
            qb_z2_b2_logvar,
            qb_z2_b2,
            pt_z2_z1_mu,
            pt_z2_z1_logvar,
            pd_x2_z2,
        )

    def calculate_loss(
        self, forward_ret: torch.Tensor, labels: torch.Tensor = None
    ) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate the loss of the model

        Parameters
        ----------
        forward_ret : torch.Tensor
            The output tensor of the model
        labels : :torch.Tensor, optional
            The input labels tensor, by default None

        Returns
        -------
        Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            _description_
        """
        (
            x,
            t2,
            qs_z1_z2_b1_mu,
            qs_z1_z2_b1_logvar,
            pb_z1_b1_mu,
            pb_z1_b1_logvar,
            qb_z2_b2_mu,
            qb_z2_b2_logvar,
            qb_z2_b2,
            pt_z2_z1_mu,
            pt_z2_z1_logvar,
            pd_x2_z2,
        ) = forward_ret

        # replicate x multiple times
        x = x[None, ...].expand(
            self.samples_per_seq, -1, -1, -1
        )  # size: copy, bs, time, dim
        x2 = torch.gather(x, 2, t2[..., None, None].expand(-1, -1, -1, x.size(3))).view(
            -1, x.size(3)
        )
        batch_size = x2.size(0)

        kl_div_qs_pb = self.kl_div_gaussian(
            qs_z1_z2_b1_mu, qs_z1_z2_b1_logvar, pb_z1_b1_mu, pb_z1_b1_logvar
        ).mean()

        kl_shift_qb_pt = (
            self.gaussian_log_prob(qb_z2_b2_mu, qb_z2_b2_logvar, qb_z2_b2)
            - self.gaussian_log_prob(pt_z2_z1_mu, pt_z2_z1_logvar, qb_z2_b2)
        ).mean()

        bce = F.binary_cross_entropy(pd_x2_z2, x2, reduction="sum") / batch_size
        bce_optimal = (
            F.binary_cross_entropy(x2, x2, reduction="sum").detach() / batch_size
        )
        bce_diff = bce - bce_optimal

        loss = bce_diff + kl_div_qs_pb + kl_shift_qb_pt

        return loss, bce_diff, kl_div_qs_pb, kl_shift_qb_pt, bce_optimal

    def rollout(
        self, x: torch.tensor, t: int, n: int, z_rollout: bool = False
    ) -> torch.tensor:
        """
        Rollout the model for the given input

        Parameters
        ----------
        x : torch.tensor
            The input tensor of an image
        t : int
            The time of the input tensor
        n : int
            The number of samples
        z_rollout : bool, optional
            If true return the latent space instead of the image rollout tensor, by default False

        Returns
        -------
        torch.tensor
            The output tensor of the model
            Either the image rollout tensor or the latent space
        """
        # pre-process image x
        processed_x = self.process_x(x)  # x length is t + 1

        # aggregate the belief b
        b = self.b_rnn(processed_x)[:, t]  # size: bs, time, layers, dim

        # compute z from b
        p_z_bs = []
        for layer in range(self.layers - 1, -1, -1):
            if layer == self.layers - 1:
                p_z_b_mu, p_z_b_logvar = self.z_b[layer](b[:, layer])
            else:
                p_z_b_mu, p_z_b_logvar = self.z_b[layer](
                    torch.cat([b[:, layer], p_z_b], dim=1)
                )
            p_z_b = self.reparameterize_gaussian(p_z_b_mu, p_z_b_logvar, True)
            p_z_bs.insert(0, p_z_b)

        z = torch.cat(p_z_bs, dim=1)
        rollout_x = []

        for _ in range(n):
            next_z = []
            for layer in range(self.layers - 1, -1, -1):  # TODO optionally condition n
                if layer == self.layers - 1:
                    pt_z2_z1_mu, pt_z2_z1_logvar = self.z2_z1[layer](z)
                else:
                    pt_z2_z1_mu, pt_z2_z1_logvar = self.z2_z1[layer](
                        torch.cat([z, pt_z2_z1], dim=1)
                    )
                pt_z2_z1 = self.reparameterize_gaussian(
                    pt_z2_z1_mu, pt_z2_z1_logvar, True
                )
                next_z.insert(0, pt_z2_z1)

            z = torch.cat(next_z, dim=1)
            rollout_x.append(self.x_z(z))

        if z_rollout:
            return z
        else:
            return torch.stack(rollout_x, dim=1)
