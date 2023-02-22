__author__ = "Keenan Manpearl"
__date__ = "2023/1/24"

"""
original code by Xinqiang Ding <xqding@umich.edu>
classes and functions needed to train model
"""

import numpy as np
import torch
import torch.nn as nn


class DBlock(nn.Module):
    """A basic building block for parameterizing a normal distribution.
    It corresponds to the D operation in the reference Appendix.

    Returns a mean and log sigma for any arbitrary context X
    [mu, log_sigma] = W3*tanh(W1*x + B1)*sigmoid(W2*x + B2) + B3
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(DBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Construct fully connected linear layers to map
        # X to a normal distribution
        self.weight1 = nn.Linear(input_size, hidden_size)
        self.weight2 = nn.Linear(input_size, hidden_size)
        self.mu = nn.Linear(hidden_size, output_size)
        self.logsigma = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        t = torch.tanh(self.weight1(input))
        t = t * torch.sigmoid(self.weight2(input))
        mu = self.mu(t)
        logsigma = self.log_sigma(t)
        return mu, logsigma


class PreProcess(nn.Module):
    """
    The pre-process layer for MNIST image
    """

    def __init__(self, input_size, processed_x_size):
        super(PreProcess, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, processed_x_size)
        self.fc2 = nn.Linear(processed_x_size, processed_x_size)

    def forward(self, input):
        t = torch.relu(self.fc1(input))
        t = torch.relu(self.fc2(t))
        return t


class Decoder(nn.Module):
    """
    The decoder layer converting state to observation.
    Because the observation is MNIST image whose elements are values
    between 0 and 1, the output of this layer are probabilities of
    elements being 1.
    """

    def __init__(self, z_size, hidden_size, x_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(in_features=z_size, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.fc3 = nn.Linear(in_features=hidden_size, out_features=x_size)

    def forward(self, z):
        t = torch.tanh(self.fc1(z))
        t = torch.tanh(self.fc2(t))
        p = torch.sigmoid(self.fc3(t))
        return p


class TD_VAE(nn.Module):
    """
    The full TD_VAE model.

    -------------------
    First, let's first go through some definitions which would help understanding what is going
    on in the following code.

    Belief: As the model is fed a sequence of observations, x_t, the model updates its belief
        state, b_t, through an LSTM network. It is a deterministic function of x_t.

    State: The latent state variable, z.

    Observation: The observed variable, x.

    -------------------
    The TD_VAE contains several interconnected layers, which all have different purposes.
    We provide more details on each layer below, but, in summary, the layers are as follows:

    1) The preprocessing layer. This converts the input images to a form ready for LSTM training
    2) The LSTM network. This network computes the belief states of the preprocessed data X
    3) The encoder network. This is a two layer network that learns a compressed representation of
        the belief states. This compressed representation is called z, and the network is called P_B.
    4) The smoothing network. This is a two layer network that learns how to imagine/hallucinate/sample
        a compressed representation z from a previous time.
    5) The transition network. Also called state prediction network or forward model. This is a two layer
        network that enables us to predict compressed representation z in the future
    6) The decoder network. This layer will learn how to reverse the compression process and convert z
        back to the original input dimensions X

    -------------------
    The TD-VAE model learns by minimizing reconstruction loss (reconstructing compressed belief state
    back to original X dimensions) and mimizing two distinct KL divergence terms. To describe these terms,
    it is helpful to realize that there are three ways to arrive at a compressed representation z.

    A) Encoder network (P_B)
    B) Smoothing network (q_s)
    C) Transition network (P_T)

    The first KL term used to learn is: KL( P_B | q_s )
    This learns if the world is providing enough information to predict an earlier compressed state.

    The second KL term used to learn is: KL( P_B | P_T ), which is learned via sampling (no closed solution)
    This learns if the future compressed states can be predicted from past information.

    Therefore, the objective function for the TD-VAE is as follows:

            loss = ||X - decoder(P_B(X))|| + KL( P_B | q_s ) + KL( P_B | P_T )

    -------------------
    We are primarily interested in two aspects of this model.

    1) We would like to be able to predict future states of the world, perhaps from static images.
    2) We would like to explore the encoded representation z formed from the belief state encoder.
    We will compare z across different contexts, which will tell us how different contexts impact
    behavior in the world over time.
    """

    def __init__(
        self,
        x_size,
        processed_x_size,
        b_size,
        z_size,
        d_block_hidden_size,
        decoder_hidden_size,
    ):
        super(TD_VAE, self).__init__()
        self.x_size = x_size
        self.processed_x_size = processed_x_size
        self.b_size = b_size
        self.z_size = z_size
        self.d_block_hidden_size = d_block_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        ##############################
        ############ Initialize layers
        ##############################
        # 1) Pre-process layer
        self.process_x = PreProcess(
            input_size=self.x_size, processed_x_size=self.processed_x_size
        )

        # 2) One layer LSTM for aggregating belief states
        self.lstm = nn.LSTM(
            input_size=self.processed_x_size, hidden_size=self.b_size, batch_first=True
        )

        ##############################
        ######### Layers in Appendix D
        ##############################
        # Note: Weights shared across time but not across layers.

        #########
        # Layers 3 and 4 accomplish sampling of compressed state z.
        # This is learning the belief distribution (P_B)
        #########

        # 3) Second layer DBlock compressing belief state
        # e.g. For compressing belief at time 2 (b_t2) to infer z at time 2 from layer 2
        # This layer is used in line 1 of Appendix D
        # Input: the belief state
        # Output: the layer 2 encoded representation (compressed representation) z
        # Remember, b is output from the LSTM
        self.encoder_b_to_z_layer2 = DBlock(
            input_size=self.b_size,
            hidden_size=self.d_block_hidden_size,
            output_size=self.z_size,
        )

        # 4) First layer DBlock compressing belief state
        # This layer is used in line 2 of Appendix D
        # Input: the belief state and also the output from the previous layer (encoder_b_to_z_layer2)
        # Output: the full encoded representation (compressed representation) z
        self.encoder_b_to_z_layer1 = DBlock(
            input_size=self.b_size + self.z_size,
            hidden_size=self.d_block_hidden_size,
            output_size=self.z_size,
        )

        #########
        # Layers 5 and 6 sample (or imagine/hallucinate) a compressed state z at a previous time (e.g. time 1).
        # The sampling uses information about the prior belief state (e.g. time 1) and the current
        # compressed state z (e.g. time 2). In other words, for example, given the compressed state at
        # time 2 (z_t2) and computed belief at time 1 (b_t1), sample a compressed state at time 1 (z_t1).
        #
        # These layers learn the smoothing distribution (q_s), which we use as a second way of computing a
        # compressed representation z, but at an earlier time point (e.g. time 1). This can be considered an
        # "ex post guess of the state of the world", or, indirectly examining how much the world can be
        # predicted. Later, when we calculate the loss (see calculate_loss), this smoothing distribution is
        # learned by trying to mimic the encoder distribution, and can therefore be viewed as a pseudo cycle-loss.
        #########

        # 5) Second layer DBlock compressing imagined z at time 2 and belief state at time 1.
        # Taking as input b_t1 and z_t2 (where z_t2 = [z_t2_layer1, z_t2_layer2])
        # This is line 5 of Appendix D.
        # Because z_t2 is two layers concatenated, multiply z_size by 2 below
        # Input: the belief state at past time and compressed state at current time
        # Output: the layer 2 encoded representation (compressed representation) z (alternate calculation to encoding)
        self.smooth_previous_z_layer2 = DBlock(
            input_size=(self.b_size + (2 * self.z_size)),
            hidden_size=self.d_block_hidden_size,
            output_size=self.z_size,
        )

        # 6) First layer DBlock compressing imaged z at time 2, belief state at time 1, and
        # compressed state z at time 1 from layer 1. In other words, this layer takes as input:
        # b_t1, z_t2, and the previous layer z_t1_layer2 (z_time1_layer2_dblock)
        # This is line 7 of Appendix D.
        # Because z_t2 is two layers concatenated, multiply z_size by 2 below, also
        # add a z_size because z_t1_layer1 needs info from z_t2_layer2
        # Input: the belief state at past time, compressed state at current time, and smoothing layer 1
        # Output: the full encoded representation (compressed representation) z (alternate calculation to encoding)
        self.smooth_previous_z_layer1 = DBlock(
            input_size=(self.b_size + (2 * self.z_size) + self.z_size),
            hidden_size=self.d_block_hidden_size,
            output_size=self.z_size,
        )

        #########
        # Layers 7 and 8 represent the forward pass of learning how the world evolves.
        # Given the compressed state of the world imagined at a previous time from the smoothing distribution,
        # predict how the state will change at time 2.
        #
        # These layers learn the transition distribution (P_T). This nework enables us to make predictions, and in
        # a "jumpy" fashion (which means far into the future, not just step by step)
        #########

        # 7) Second layer DBlock compressing imagined state z at time 1.
        # This represents line 9 in Appendix D
        # Because z_time1 is two layers of z, multiply z_size by 2 below
        # Input: compressed representation z
        # Output: Layer 2 prediction of future state z
        self.transition_z_layer2 = DBlock(
            input_size=2 * self.z_size,
            hidden_size=self.d_block_hidden_size,
            output_size=self.z_size,
        )

        # 8) First layer DBlock compressing imagined state z at time 1. For sampling how world evolves
        # This represents line 10 in Appendix D
        # Because z_time1 is two layers of z, multiply z_size by 2 below and add z_size to incoporate
        # prevous layer
        # Input: compressed representation z, and transition network layer 2
        # Output: Full prediction of future state z
        self.transition_z_layer1 = DBlock(
            input_size=(2 * self.z_size) + self.z_size,
            hidden_size=self.d_block_hidden_size,
            output_size=self.z_size,
        )

        #########
        # Layer 9 represents the decoder, which decodes the compressed state into the observed state.
        #
        # This layer enables us to learn through a reconstruction loss and also will fully enable us to
        # sample future predictions.
        #########

        # 9) Decoder from state z to observation x
        # This represents line 11 in Appendix D
        # Because z is two layers, multiply z_size by 2 below
        # Input: Compressed representation z
        # Output: X_hat (estimated X) of the original input data dimensions (per frame only)
        self.decoder_z_to_x = Decoder(
            z_size=2 * self.z_size,
            hidden_size=self.decoder_hidden_size,
            x_size=self.x_size,
        )

    def forward(self, images):
        self.batch_size = images.size()[0]
        self.x = images

        # Pre-precess image x
        self.processed_x = self.process_x(self.x)

        # Aggregate the belief b - the model only uses b in subsequent steps
        self.b, (self.h_n, self.c_n) = self.lstm(self.processed_x)

    def calculate_loss(self, t1, t2):
        """
        Calculate the VD-VAE loss, which corresponds to equations (6) and (8) in the TD-VAE paper.

        We provide more explicit details below, but, in summary, the loss contains three core elements:

        1) The difference between the encoded z at time 1 and the hallucinated (or imagined) z at time 1

            The encoded z at time 1 is generated two ways: 1) the belief distribution (P_B) and 2) the
            smoothing distribution (q_s). The smoothing distribution learns the compressed z at time 2 from
            information of compressed z at time 1 and belief state at time 2. This loss term ensures that
            the hallucination of future data looks like how future data might be expected to behave given
            certain states of the world that were previously observed. Importantly, we can sample from the
            belief distribution to generate new data.

                This term enables data generation.

        2) The difference between the encoded z at time 2 and the predicted z at time 2

            The encoded z at time 2 is generated two ways as well: 1) the belief distribution (P_B), which
            is the same way we encode z at time 1 above, and 2) the state prediction network (or forward model).
            This loss term ensures that we're able to predict the compressed state of the world in a jumpy manner
            (meaning, more than one step in the future) and that this prediction is similar to our encoder model.
            Importantly, we can make predictions about future states with the transition network.

                This term enables state prediction.

        3) The reconstruction term between the decoded z at time 2 and real data at time 2

            This is a standard reconstruction term. We pass input data x through an encoder (belief network) to
            a lower dimensional compressed z. We then decode this z through a decoder to obtain the same input
            dimensions as X. We minimize the difference between this decoded X and the ground truth X. Importantly,
            with this decoder we can now fully simulate new data in the original dimension.

                This term fully enables data generation (original dimension).
        """

        ############################
        # Prior to calculating the loss, sample from the previously defined layers.
        # We will use these layers in subsequent components of the loss function.
        ############################
        # 1) compress z at time 2 using information from belief state at time 2.
        (
            z_time2_layer2_mu,
            z_time2_layer2_logsigma,
        ) = self.encoder_b_to_z_layer2(self.b[:, t2, :])

        z_time2_layer2_epsilon = torch.randn_like(
            z_time2_layer2_mu
        )  # Note, this is the reparameterization

        z_time2_layer2 = (
            z_time2_layer2_mu
            + torch.exp(z_time2_layer2_logsigma) * z_time2_layer2_epsilon
        )

        # Sample z at time 2 from the belief state at time 2 (layer 1)
        (z_time2_layer1_mu, z_time2_layer1_logsigma,) = self.encoder_b_to_z_layer1(
            torch.cat((self.b[:, t2, :], z_time2_layer2), dim=-1)
        )

        z_time2_layer2_epsilon = torch.randn_like(z_time2_layer1_mu)

        z_time2_layer1 = (
            z_time2_layer1_mu
            + torch.exp(z_time2_layer1_logsigma) * z_time2_layer2_epsilon
        )

        # Concatenate z from layer 1 and layer 2
        z_time2 = torch.cat((z_time2_layer1, z_time2_layer2), dim=-1)

        # 2) Imagine z at time 1 given belief at time 1 and z at time 2 (hallucinate z_time1)
        # This is the smoothing distribution (q_s)
        (
            z_time1_layer2_smoothing_mu,
            z_time1_layer2_smoothing_logsigma,
        ) = self.smooth_previous_z_layer2(
            torch.cat((self.b[:, t1, :], z_time2), dim=-1)
        )
        z_time1_layer2_smoothing_epsilon = torch.randn_like(z_time1_layer2_smoothing_mu)
        z_time1_layer2_smoothing = (
            z_time1_layer2_smoothing_mu
            + torch.exp(z_time1_layer2_smoothing_logsigma)
            * z_time1_layer2_smoothing_epsilon
        )

        (
            z_time1_layer1_smoothing_mu,
            z_time1_layer1_smoothing_logsigma,
        ) = self.smooth_previous_z_layer1(
            torch.cat((self.b[:, t1, :], z_time2, z_time1_layer2_smoothing), dim=-1)
        )
        z_time1_layer1_smoothing_epsilon = torch.randn_like(z_time1_layer1_smoothing_mu)
        z_time1_layer1_smoothing = (
            z_time1_layer1_smoothing_mu
            + torch.exp(z_time1_layer1_smoothing_logsigma)
            * z_time1_layer1_smoothing_epsilon
        )

        z_time1_smoothing = torch.cat(
            (z_time1_layer1_smoothing, z_time1_layer2_smoothing), dim=-1
        )

        # 3) Obtain the actual compressed z at time 1
        # This is the belief distribution (P_B)
        # Note: Use the layer trained to compress belief at time 2 into z to compress belief at time 1
        (
            z_time1_layer2_belief_mu,
            z_time1_layer2_belief_logsigma,
        ) = self.encoder_b_to_z_layer2(self.b[:, t1, :])
        (
            z_time1_layer1_belief_mu,
            z_time1_layer1_belief_logsigma,
        ) = self.encoder_b_to_z_layer1(
            torch.cat((self.b[:, t1, :], z_time1_layer2_smoothing), dim=-1)
        )  # Note, only use layer 2 smoothing here as a bottleneck to not encode too much information

        # 4) Forward pass of how the world evolves
        # This is the transition distribution (P_T)
        (
            z_time2_layer2_transition_mu,
            z_time2_layer2_transition_logsigma,
        ) = self.transition_z_layer2(z_time1_smoothing)
        (
            z_time2_layer1_transition_mu,
            z_time2_layer1_transition_logsigma,
        ) = self.transition_z_layer1(
            torch.cat((z_time1_smoothing, z_time2_layer2_transition_mu), dim=-1)
        )

        # 5) Decode the predicted state z at time 2 into real observation (Decoder)
        # Note: This is where you "Ground the state in observation"
        x_time2_decoded = self.decoder_z_to_x(z_time2)

        ############################
        # Now that we have computed some key variables, we can process the loss function.
        # The loss function contains five terms, which funnel information
        # from various parts of the model. The loss is as follows:
        # 1) Minimize difference between ground truth z_time1 (P_B) and hallucinated z_time1 (q_s)
        #    Note, only using layer 2
        # 2) Minimize difference between ground truth z_time1 (P_B) and hallucinated z_time1 (q_s)
        #    Note, only using layer 1
        # 3) Minimize difference between ground truth z_time2 (P_B) and forward pass z_time2 (P_T)
        #    Note, only using layer 2
        # 4) Minimize difference between ground truth z_time2 (P_B) and forward pass z_time2 (P_T)
        #    Note, only using layer 1
        # 5) Minimize the difference between decoder and ground truth observation (reconstruction)
        #
        # We will walk through calculating each of these five components.
        ############################

        # 1) KL divergence between belief distribution at time 1 and smoothing distribution at time 1
        # This is line 12 of Appendix D representing loss_time1_layer2 (KL(q_s_time1_layer2 | belief_time1_layer2))
        # Note: this uses information from layer 2 only
        loss_kl_belief_smoothing_z_time1_layer2 = (
            0.5
            * torch.sum(
                (
                    (z_time1_layer2_belief_mu - z_time1_layer2_smoothing)
                    / torch.exp(z_time1_layer2_belief_logsigma)
                )
                ** 2,
                -1,
            )
            + torch.sum(z_time1_layer2_belief_logsigma, -1)
            - torch.sum(z_time1_layer2_smoothing_logsigma, -1)
        )

        # 2) KL divergence between belief distribution at time 1 and smoothing distribution at time 1
        # This is line 13 of Appendix D representing loss_time1_layer1 (KL(q_s_time1_layer1 | belief_time1_layer1))
        # Note: this uses information from layer 1 only
        loss_kl_belief_smoothing_z_time1_layer1 = (
            0.5
            * torch.sum(
                (
                    (z_time1_layer1_belief_mu - z_time1_layer1_smoothing)
                    / torch.exp(z_time1_layer2_belief_logsigma)
                )
                ** 2,
                -1,
            )
            + torch.sum(z_time1_layer1_belief_logsigma, -1)
            - torch.sum(z_time1_layer1_smoothing_logsigma, -1)
        )

        #########
        # Note: The following four terms estimate the KL divergence between z at time t2 based on
        # variational distribution (inference model) and z at time t2 based on transition. In contrast
        # with the above KL divergence for z distribution at time t1, this KL divergence can not be calculated
        # analytically because the transition distribution depends on z_t1, which is sampled after z_t2.
        # Therefore, the KL divergence is estimated by sampling.
        #########
        # 3) KL divergence between belief distribution at time 2 and transition distribution at time 2
        # This is line 14 of Appendix D representing loss_time2_layer2 (log(P_B(z_time2)) - log(P_T(z_time2)))
        loss_kl_belief_transition_z_time2_layer2 = torch.sum(
            -0.5 * z_time2_layer2_epsilon**2
            - 0.5
            * z_time2_layer2_epsilon.new_tensor(
                2 * np.pi
            )  # A random sample from the reparameterization
            - z_time2_layer2_logsigma,
            dim=-1,
        )

        # Note that we're subtracting the previous sampling from this KL term
        loss_kl_belief_transition_z_time2_layer2 += torch.sum(
            0.5
            * (
                (z_time2_layer2 - z_time2_layer2_transition_mu)
                / torch.exp(z_time2_layer2_transition_log_sigma)
            )
            ** 2
            + 0.5 * z_time2_layer2.new_tensor(2 * np.pi)
            + z_time2_layer2_transition_log_sigma,
            -1,
        )

        # 4) KL divergence between belief distribution at time 2 and transition distribution at time 2
        # This is line 15 of Appendix D representing loss_time2_layer1 (log(P_B(z_time2)) - log(P_T(z_time2)))
        loss_kl_belief_transition_z_time2_layer1 = torch.sum(
            -0.5 * z_time2_layer1_epsilon**2
            - 0.5 * z_time2_layer1_epsilon.new_tensor(2 * np.pi)
            - z_time2_layer1_logsigma,
            dim=-1,
        )

        # Note that we're subtracting the previous sampling from this KL term
        loss_kl_belief_transition_z_time2_layer1 += torch.sum(
            0.5
            * (
                (z_time2_layer1 - z_time2_layer1_transition_mu)
                / torch.exp(z_time2_layer1_transition_logsigma)
            )
            ** 2
            + 0.5 * z_time2_layer1.new_tensor(2 * np.pi)
            + z_time2_layer1_transition_logsigma,
            -1,
        )

        # 5) Reconstruction loss (calculate difference between decoded z at time 2 and ground truth observation at time 2)
        # This is line 16 of Appendix D
        reconstruction_loss = -torch.sum(
            self.x[:, t2, :] * torch.log(x_time2_decoded)
            + (1 - self.x[:, t2, :]) * torch.log(1 - x_time2_decoded),
            -1,
        )

        # Compile all loss terms and take the mean
        loss = (
            loss_kl_belief_smoothing_z_time1_layer2
            + loss_kl_belief_smoothing_z_time1_layer1
            + loss_kl_belief_transition_z_time2_layer2
            + loss_kl_belief_transition_z_time2_layer1
            + reconstruction_loss
        )
        loss = torch.mean(loss)

        return loss

    def rollout(self, images, t1, t2):
        # Preprocess images and pass through LSTM
        self.forward(images)

        # At time t1-1, we encode a state z based on belief at time t1-1
        z_layer2_mu, z_layer2_logsigma = self.encoder_b_to_z_layer2(
            self.b[:, t1 - 1, :]
        )
        z_layer2_epsilon = torch.randn_like(z_layer2_mu)
        z_layer2 = z_layer2_mu + torch.exp(z_layer2_logsigma) * z_layer2_epsilon

        z_layer1_mu, z_layer1_logsigma = self.encoder_b_to_z_layer1(
            torch.cat((self.b[:, t1 - 1, :], z_layer2), dim=-1)
        )
        z_layer1_epsilon = torch.randn_like(z_layer1_mu)
        z_layer1 = z_layer1_mu + torch.exp(z_layer1_logsigma) * z_layer1_epsilon

        z = torch.cat((z_layer1, z_layer2), dim=-1)

        rollout_x = []
        for k in range(t2 - t1 + 1):
            # Predict states after time t1 using state transition network
            (
                predict_z_layer2_mu,
                predict_z_layer2_logsigma,
            ) = self.transition_z_layer2(z)
            predict_z_layer2_epsilon = torch.randn_like(predict_z_layer2_mu)
            predict_z_layer2 = (
                predict_z_layer2_mu
                + torch.exp(predict_z_layer2_logsigma) * predict_z_layer2_epsilon
            )

            predict_z_layer1_mu, predict_z_layer1_logsigma = self.l1_transition_z(
                torch.cat((z, predict_z_layer2), dim=-1)
            )
            predict_z_layer1_epsilon = torch.randn_like(predict_z_layer1_mu)
            predict_z_layer1 = (
                predict_z_layer1_mu
                + torch.exp(predict_z_layer1_logsigma) * predict_z_layer1_epsilon
            )

            predict_z = torch.cat((predict_z_layer1, predict_z_layer2), dim=-1)

            # Decode sampled state z_t1 to predict x
            predict_x = self.decoder_z_to_x(predict_z)
            rollout_x.append(predict_x)

            z = predict_z

        return rollout_x
