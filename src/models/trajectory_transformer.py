from abc import abstractmethod
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from gymnasium.spaces import Box, Dict
from torchtyping import TensorType as TT
from transformer_lens import HookedTransformer, HookedTransformerConfig

from src.config import EnvironmentConfig, TransformerModelConfig

from .components import (
    MiniGridConvEmbedder,
    PosEmbedTokens,
    MiniGridViTEmbedder,
)
import transformers
from src.models.trajectory_gpt2 import GPT2Model

class TrajectoryTransformer(nn.Module):
    """
    Base Class for trajectory modelling transformers including:
        - Decision Transformer (offline, RTG, (R,s,a))
        - Online Transformer (online, reward, (s,a,r) or (s,a))
    """

    def __init__(
        self,
        transformer_config: TransformerModelConfig,
        environment_config: EnvironmentConfig,
    ):
        super().__init__()

        self.transformer_config = transformer_config
        self.environment_config = environment_config

        # Why is this in a sequential? Need to get rid of it at some
        # point when I don't care about loading older models.
        self.action_embedding = nn.Sequential(
            nn.Embedding(
                environment_config.action_space.n + 1,
                self.transformer_config.d_model,
            )
        )
        self.time_embedding = self.initialize_time_embedding()
        self.state_embedding = self.initialize_state_embedding()

        # Initialize weights
        nn.init.normal_(
            self.action_embedding[0].weight,
            mean=0.0,
            std=1
            / (
                (environment_config.action_space.n + 1 + 1)
                * self.transformer_config.d_model
            ),
        )

        self.transformer = self.initialize_easy_transformer()

        self.action_predictor = nn.Linear(
            self.transformer_config.d_model, environment_config.action_space.n
        )
        self.initialize_state_predictor()

        self.initialize_weights()

    def initialize_weights(self):
        """
        TransformerLens is weird so we have to use the module path
        and can't just rely on the module instance as we do would
        be the default approach in pytorch.
        """
        self.apply(self._init_weights_classic)

        for name, param in self.named_parameters():
            if "W_" in name:
                nn.init.normal_(param, std=0.02)

    def _init_weights_classic(self, module):
        """
        Use Min GPT Method.
        https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/model.py#L163

        Will need to check that this works with the transformer_lens library.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif (
            "PosEmbedTokens" in module._get_name()
        ):  # transformer lens components
            for param in module.parameters():
                torch.nn.init.normal_(param, mean=0.0, std=0.02)

    def get_time_embedding(self, timesteps):
        assert (
            timesteps.max() <= self.environment_config.max_steps
        ), "timesteps must be less than max_timesteps"

        block_size = timesteps.shape[1]
        timesteps = rearrange(
            timesteps, "batch block time-> (batch block) time"
        )
        time_embeddings = self.time_embedding(timesteps)
        if self.transformer_config.time_embedding_type != "linear":
            time_embeddings = time_embeddings.squeeze(-2)
        time_embeddings = rearrange(
            time_embeddings,
            "(batch block) n_embd -> batch block n_embd",
            block=block_size,
        )
        return time_embeddings

    def get_state_embedding(self, states):
        # embed states and recast back to (batch, block_size, n_embd)
        block_size = states.shape[1]
        if self.transformer_config.state_embedding_type.lower() in [
            "cnn",
            "vit",
        ]:
            states = rearrange(
                states,
                "batch block height width channel -> (batch block) height width channel",
            )
            state_embeddings = self.state_embedding(
                states.type(torch.float32).contiguous()
            )  # (batch * block_size, n_embd)

        elif self.transformer_config.state_embedding_type.lower() == "grid":
            states = rearrange(
                states,
                "batch block height width channel -> (batch block) (channel height width)",
            )
            state_embeddings = self.state_embedding(
                states.type(torch.float32).contiguous()
            )  # (batch * block_size, n_embd)
        else:
            states = rearrange(
                states, "batch block state_dim -> (batch block) state_dim"
            )
            state_embeddings = self.state_embedding(
                states.type(torch.float32).contiguous()
            )
        state_embeddings = rearrange(
            state_embeddings,
            "(batch block) n_embd -> batch block n_embd",
            block=block_size,
        )
        return state_embeddings

    def get_action_embedding(self, actions):
        block_size = actions.shape[1]
        if block_size == 0:
            return None  # no actions to embed
        actions = rearrange(
            actions, "batch block action -> (batch block) action"
        )
        # I don't see why we need this but we do? Maybe because of the sequential?
        action_embeddings = self.action_embedding(actions).flatten(1)
        action_embeddings = rearrange(
            action_embeddings,
            "(batch block) n_embd -> batch block n_embd",
            block=block_size,
        )
        return action_embeddings

    def predict_states(self, x):
        return self.state_predictor(x)

    def predict_actions(self, x):
        return self.action_predictor(x)

    @abstractmethod
    def get_token_embeddings(
        self, state_embeddings, time_embeddings, action_embeddings, **kwargs
    ):
        """
        Returns the token embeddings for the transformer input.
        Note that different subclasses will have different token embeddings
        such as the DecisionTransformer which will use RTG (placed before the
        state embedding).

        Args:
            states: (batch, position, state_dim)
            actions: (batch, position)
            timesteps: (batch, position)
        Kwargs:
            rtgs: (batch, position) (only for DecisionTransformer)

        Returns:
            token_embeddings: (batch, position, n_embd)
        """
        pass

    @abstractmethod
    def get_action(self, **kwargs) -> int:
        """
        Returns the action given the state.
        """
        pass

    def initialize_time_embedding(self):
        if not (self.transformer_config.time_embedding_type == "linear"):
            self.time_embedding = nn.Embedding(
                self.environment_config.max_steps + 1,
                self.transformer_config.d_model,
            )
        else:
            self.time_embedding = nn.Linear(1, self.transformer_config.d_model)

        return self.time_embedding

    def initialize_state_embedding(self):
        if self.transformer_config.state_embedding_type.lower() == "cnn":
            state_embedding = MiniGridConvEmbedder(
                self.transformer_config.d_model, endpool=True
            )
        elif self.transformer_config.state_embedding_type.lower() == "vit":
            state_embedding = MiniGridViTEmbedder(
                self.transformer_config.d_model,
            )
        else:
            if isinstance(self.environment_config.observation_space, Dict):
                n_obs = np.prod(
                    self.environment_config.observation_space["image"].shape
                )
            else:
                n_obs = np.prod(
                    self.environment_config.observation_space.shape
                )

            state_embedding = nn.Linear(
                n_obs, self.transformer_config.d_model, bias=False
            )

            nn.init.normal_(state_embedding.weight, mean=0.0, std=0.02)

        return state_embedding

    def initialize_state_predictor(self):
        if isinstance(self.environment_config.observation_space, Box):
            self.state_predictor = nn.Linear(
                self.transformer_config.d_model,
                np.prod(self.environment_config.observation_space.shape),
            )
        elif isinstance(self.environment_config.observation_space, Dict):
            self.state_predictor = nn.Linear(
                self.transformer_config.d_model,
                np.prod(
                    self.environment_config.observation_space["image"].shape
                ),
            )

    def initialize_easy_transformer(self):
        # Transformer
        cfg = HookedTransformerConfig(
            n_layers=self.transformer_config.n_layers,
            d_model=self.transformer_config.d_model,
            d_head=self.transformer_config.d_head,
            n_heads=self.transformer_config.n_heads,
            d_mlp=self.transformer_config.d_mlp,
            d_vocab=self.transformer_config.d_model,
            # 3x the max timestep so we have room for an action, reward, and state per timestep
            n_ctx=self.transformer_config.n_ctx,
            act_fn=self.transformer_config.activation_fn,
            gated_mlp=self.transformer_config.gated_mlp,
            normalization_type=self.transformer_config.layer_norm,
            attention_dir="causal",
            d_vocab_out=self.transformer_config.d_model,
            seed=self.transformer_config.seed,
            device=self.transformer_config.device,
        )

        assert (
            cfg.attention_dir == "causal"
        ), "Attention direction must be causal"
        # assert cfg.normalization_type is None, "Normalization type must be None"

        transformer = HookedTransformer(cfg)

        # Because we passing in tokens, turn off embedding and update the position embedding
        transformer.embed = nn.Identity()
        transformer.pos_embed = PosEmbedTokens(cfg)
        # initialize position embedding
        nn.init.normal_(transformer.pos_embed.W_pos, cfg.initializer_range)
        # don't unembed, we'll do that ourselves.
        transformer.unembed = nn.Identity()

        return transformer


class DecisionTransformer(TrajectoryTransformer):
    def __init__(self, environment_config, transformer_config, **kwargs):
        super().__init__(
            environment_config=environment_config,
            transformer_config=transformer_config,
            **kwargs,
        )
        self.model_type = "decision_transformer"
        self.reward_embedding = nn.Sequential(
            nn.Linear(1, self.transformer_config.d_model, bias=False)
        )
        self.reward_predictor = nn.Linear(self.transformer_config.d_model, 1)

        # n_ctx include full timesteps except for the last where it doesn't know the action
        assert (transformer_config.n_ctx - 2) % 3 == 0

        self.initialize_weights()

    def predict_rewards(self, x):
        return self.reward_predictor(x)

    def get_token_embeddings(
        self,
        state_embeddings,
        time_embeddings,
        reward_embeddings,
        action_embeddings=None,
        targets=None,
    ):
        """
        We need to compose the embeddings for:
            - states
            - actions
            - rewards
            - time

        Handling the cases where:
        1. we are training:
            1. we may not have action yet (reward, state)
            2. we have (action, state, reward)...
        2. we are evaluating:
            1. we have a target "a reward" followed by state

        1.1 and 2.1 are the same, but we need to handle the target as the initial reward.

        """
        batches = state_embeddings.shape[0]
        timesteps = time_embeddings.shape[1]

        reward_embeddings = reward_embeddings + time_embeddings
        state_embeddings = state_embeddings + time_embeddings

        if action_embeddings is not None:
            if action_embeddings.shape[1] < timesteps:
                assert (
                    action_embeddings.shape[1] == timesteps - 1
                ), "Action embeddings must be one timestep less than state embeddings"
                action_embeddings = (
                    action_embeddings
                    + time_embeddings[:, : action_embeddings.shape[1]]
                )
                trajectory_length = timesteps * 3 - 1
            else:
                action_embeddings = action_embeddings + time_embeddings
                trajectory_length = timesteps * 3
        else:
            trajectory_length = 2  # one timestep, no action yet

        if targets:
            targets = targets + time_embeddings

        # create the token embeddings
        token_embeddings = torch.zeros(
            (batches, trajectory_length, self.transformer_config.d_model),
            dtype=torch.float32,
            device=state_embeddings.device,
        )  # batches, blocksize, n_embd

        if action_embeddings is not None:
            token_embeddings[:, ::3, :] = reward_embeddings
            token_embeddings[:, 1::3, :] = state_embeddings
            token_embeddings[:, 2::3, :] = action_embeddings
        else:
            token_embeddings[:, 0, :] = reward_embeddings[:, 0, :]
            token_embeddings[:, 1, :] = state_embeddings[:, 0, :]

        if targets is not None:
            target_embedding = self.reward_embedding(targets)
            token_embeddings[:, 0, :] = target_embedding[:, 0, :]

        return token_embeddings

    def to_tokens(self, states, actions, rtgs, timesteps):
        # embed states and recast back to (batch, block_size, n_embd)
        state_embeddings = self.get_state_embedding(
            states
        )  # batch_size, block_size, n_embd
        action_embeddings = (
            self.get_action_embedding(actions) if actions is not None else None
        )  # batch_size, block_size, n_embd or None
        reward_embeddings = self.get_reward_embedding(
            rtgs
        )  # batch_size, block_size, n_embd
        time_embeddings = self.get_time_embedding(
            timesteps
        )  # batch_size, block_size, n_embd

        # use state_embeddings, actions, rewards to go and
        token_embeddings = self.get_token_embeddings(
            state_embeddings=state_embeddings,
            action_embeddings=action_embeddings,
            reward_embeddings=reward_embeddings,
            time_embeddings=time_embeddings,
        )
        return token_embeddings

    def get_action(self, states, actions, rewards, timesteps):
        state_preds, action_preds, reward_preds = self.forward(
            states, actions, rewards, timesteps
        )

        # get the action prediction
        action_preds = action_preds[:, -1, :]  # (batch, n_actions)
        action = torch.argmax(action_preds, dim=-1)  # (batch)
        return action

    def get_reward_embedding(self, rtgs):
        block_size = rtgs.shape[1]
        rtgs = rearrange(rtgs, "batch block rtg -> (batch block) rtg")
        rtg_embeddings = self.reward_embedding(rtgs.type(torch.float32))
        rtg_embeddings = rearrange(
            rtg_embeddings,
            "(batch block) n_embd -> batch block n_embd",
            block=block_size,
        )
        return rtg_embeddings

    def get_logits(self, x, batch_size, seq_length, no_actions: bool):
        if no_actions is False:
            # TODO replace with einsum
            if (x.shape[1] % 3 != 0) and ((x.shape[1] + 1) % 3 == 0):
                x = torch.concat((x, x[:, -2].unsqueeze(1)), dim=1)

            x = x.reshape(
                batch_size, seq_length, 3, self.transformer_config.d_model
            )
            x = x.permute(0, 2, 1, 3)

            # predict next return given state and action
            reward_preds = self.predict_rewards(x[:, 2])
            # predict next state given state and action
            state_preds = self.predict_states(x[:, 2])
            # predict next action given state and RTG
            action_preds = self.predict_actions(x[:, 1])
            return state_preds, action_preds, reward_preds

        else:
            # TODO replace with einsum
            x = x.reshape(
                batch_size, seq_length, 2, self.transformer_config.d_model
            )
            x = x.permute(0, 2, 1, 3)
            # predict next action given state and RTG
            action_preds = self.predict_actions(x[:, 1])
            return None, action_preds, None

    def forward(
        self,
        # has variable shape, starting with batch, position
        states: TT[...],  # noqa: F821
        actions: TT["batch", "position"],  # noqa: F821
        rtgs: TT["batch", "position"],  # noqa: F821
        timesteps: TT["batch", "position"],  # noqa: F821
        pad_action: bool = True,
    ) -> Tuple[
        TT[...], TT["batch", "position"], TT["batch", "position"]  # noqa: F821
    ]:
        batch_size = states.shape[0]
        seq_length = states.shape[1]
        no_actions = actions is None

        if no_actions is False:
            if actions.shape[1] < seq_length - 1:
                raise ValueError(
                    f"Actions required for all timesteps except the last, got {actions.shape[1]} and {seq_length}"
                )

            # if actions.shape[1] == seq_length - 1:
            #     if pad_action:
            #         print(
            #             "Warning: actions are missing for the last timestep, padding with zeros")
            #         # This means that you can't interpret Reward or State predictions for the last timestep!!!
            #         actions = torch.cat([actions, torch.zeros(
            #             batch_size, 1, 1, dtype=torch.long, device=actions.device)], dim=1)

        # embed states and recast back to (batch, block_size, n_embd)
        token_embeddings = self.to_tokens(states, actions, rtgs, timesteps)
        x = self.transformer(token_embeddings)
        state_preds, action_preds, reward_preds = self.get_logits(
            x, batch_size, seq_length, no_actions=no_actions
        )

        return state_preds, action_preds, reward_preds


class CloneTransformer(TrajectoryTransformer):
    """
    Behavioral clone modelling transformer including:
        - CloneTransformer (offline, (s,a))
    """

    def __init__(
        self,
        transformer_config: TransformerModelConfig,
        environment_config: EnvironmentConfig,
    ):
        super().__init__(transformer_config, environment_config)
        self.model_type = "clone_transformer"
        # n_ctx must be odd (previous state, action, next state)
        assert (transformer_config.n_ctx - 1) % 2 == 0
        self.transformer = (
            self.initialize_easy_transformer()
        )  # this might not be needed?

        self.initialize_weights()

    def get_token_embeddings(
        self, state_embeddings, time_embeddings, action_embeddings=None
    ):
        """
        Returns the token embeddings for the transformer input.
        Note that different subclasses will have different token embeddings
        such as the DecisionTransformer which will use RTG (placed before the
        state embedding).

        Args:
            states: (batch, position, state_dim)
            actions: (batch, position)

        Returns:
            token_embeddings: (batch, position, n_embd)
        """
        batches = state_embeddings.shape[0]
        timesteps = time_embeddings.shape[1]

        state_embeddings = state_embeddings + time_embeddings

        if action_embeddings is not None:
            if action_embeddings.shape[1] == time_embeddings.shape[1] - 1:
                # missing action for last t-step.
                action_embeddings = action_embeddings + time_embeddings[:, :-1]
                # repeat the last action embedding for the last timestep
                action_embeddings = torch.cat(
                    [
                        action_embeddings,
                        action_embeddings[:, -1, :].unsqueeze(1),
                    ],
                    dim=1,
                )
                # now the last action and second last are duplicates but we can fix this later. (TODO)
                trajectory_length = timesteps * 2
            else:
                action_embeddings = action_embeddings + time_embeddings
                trajectory_length = timesteps * 2
        else:
            trajectory_length = 1  # one timestep, no action yet

        # create the token embeddings
        token_embeddings = torch.zeros(
            (batches, trajectory_length, self.transformer_config.d_model),
            dtype=torch.float32,
            device=state_embeddings.device,
        )  # batches, blocksize, n_embd

        if action_embeddings is not None:
            token_embeddings[:, 0::2, :] = state_embeddings
            token_embeddings[:, 1::2, :] = action_embeddings
        else:
            token_embeddings[:, 0, :] = state_embeddings[:, 0, :]

        return token_embeddings

    def to_tokens(self, states, actions, timesteps):
        # embed states and recast back to (batch, block_size, n_embd)
        state_embeddings = self.get_state_embedding(
            states
        )  # batch_size, block_size, n_embd
        action_embeddings = (
            self.get_action_embedding(actions) if actions is not None else None
        )  # batch_size, block_size, n_embd or None
        time_embeddings = self.get_time_embedding(
            timesteps
        )  # batch_size, block_size, n_embd

        # use state_embeddings, actions, rewards to go and
        token_embeddings = self.get_token_embeddings(
            state_embeddings=state_embeddings,
            action_embeddings=action_embeddings,
            time_embeddings=time_embeddings,
        )
        return token_embeddings

    def forward(
        self,
        # has variable shape, starting with batch, position
        states: TT[...],
        actions: TT["batch", "position"],  # noqa: F821
        timesteps: TT["batch", "position"],  # noqa: F821
        pad_action: bool = True,
    ) -> Tuple[
        TT[...], TT["batch", "position"], TT["batch", "position"]  # noqa: F821
    ]:
        batch_size = states.shape[0]
        seq_length = states.shape[1]

        if (
            seq_length + (seq_length - 1) * (actions is not None)
            > self.transformer_config.n_ctx
        ):
            raise ValueError(
                f"Sequence length is too long for transformer, got {seq_length} and {self.transformer_config.n_ctx}"
            )

        no_actions = (actions is None) or (actions.shape[1] == 0)

        if no_actions is False:
            if actions.shape[1] < seq_length - 1:
                raise ValueError(
                    f"Actions required for all timesteps except the last, got {actions.shape[1]} and {seq_length}"
                )

            if actions.shape[1] != seq_length - 1:
                if pad_action:
                    print(
                        "Warning: actions are missing for the last timestep, padding with zeros"
                    )
                    # This means that you can't interpret Reward or State predictions for the last timestep!!!
                    actions = torch.cat(
                        [
                            torch.zeros(
                                batch_size,
                                1,
                                1,
                                dtype=torch.long,
                                device=actions.device,
                            ),
                            actions,
                        ],
                        dim=1,
                    )

        # embed states and recast back to (batch, block_size, n_embd)
        token_embeddings = self.to_tokens(states, actions, timesteps)

        if no_actions is False:
            if actions.shape[1] == states.shape[1] - 1:
                x = self.transformer(token_embeddings[:, :-1])
                # concat last action embedding to the end of the transformer output x[:,-2].unsqueeze(1)
                x = torch.cat(
                    [x, token_embeddings[:, -2, :].unsqueeze(1)], dim=1
                )
                state_preds, action_preds = self.get_logits(
                    x, batch_size, seq_length, no_actions=no_actions
                )
            else:
                x = self.transformer(token_embeddings)
                state_preds, action_preds = self.get_logits(
                    x, batch_size, seq_length, no_actions=no_actions
                )
        else:
            x = self.transformer(token_embeddings)
            state_preds, action_preds = self.get_logits(
                x, batch_size, seq_length, no_actions=no_actions
            )

        return state_preds, action_preds

    def get_action(self, states, actions, timesteps):
        state_preds, action_preds = self.forward(states, actions, timesteps)

        # get the action prediction
        action_preds = action_preds[:, -1, :]  # (batch, n_actions)
        action = torch.argmax(action_preds, dim=-1)  # (batch)
        return action

    def get_logits(self, x, batch_size, seq_length, no_actions: bool):
        # TODO replace with einsum
        if not no_actions:
            x = x.reshape(
                batch_size, seq_length, 2, self.transformer_config.d_model
            ).permute(0, 2, 1, 3)
            # predict next return given state and action
            # reward_preds = self.predict_rewards(x[:, 2])
            # predict next state given state and action
            state_preds = self.predict_states(x[:, 1])
            # predict next action given state
            action_preds = self.predict_actions(x[:, 0])

            return state_preds, action_preds
        else:
            x = x.reshape(
                batch_size, seq_length, 1, self.transformer_config.d_model
            ).permute(0, 2, 1, 3)

            # predict next return given state and action
            # reward_preds = self.predict_rewards(x[:, 2])
            # predict next state given state and action
            # predict next action given state
            action_preds = self.predict_actions(x[:, 0])

            return None, action_preds


class ActorTransformer(CloneTransformer):
    """
    Identical to clone transformer but forward pass can only return action predictions
    """

    def __init__(
        self,
        transformer_config: TransformerModelConfig,
        environment_config: EnvironmentConfig,
    ):
        super().__init__(transformer_config, environment_config)

    def forward(
        self,
        # has variable shape, starting with batch, position
        states: TT[...],
        actions: TT["batch", "position"],  # noqa: F821
        timesteps: TT["batch", "position"],  # noqa: F821
        pad_action: bool = True,
    ) -> TT["batch", "position"]:  # noqa: F821
        _, action_preds = super().forward(
            states, actions, timesteps, pad_action=pad_action
        )

        return action_preds


class CriticTransformer(CloneTransformer):
    """
    Identical to clone transformer but forward pass can only return state predictions
    """

    def __init__(
        self,
        transformer_config: TransformerModelConfig,
        environment_config: EnvironmentConfig,
    ):
        super().__init__(transformer_config, environment_config)
        self.value_predictor = nn.Linear(
            transformer_config.d_model, 1, bias=True
        )
        self.initialize_weights()

    def forward(
        self,
        # has variable shape, starting with batch, position
        states: TT[...],
        actions: TT["batch", "position"],  # noqa: F821
        timesteps: TT["batch", "position"],  # noqa: F821
        pad_action: bool = True,
    ) -> TT[...]:  # noqa: F821
        _, value_pred = super().forward(
            states, actions, timesteps, pad_action=pad_action
        )

        return value_pred

    # hacky way to predict values instead of actions with same information
    def predict_actions(self, x):
        return self.value_predictor(x)

# Decision Transformer model from original
class TrajT(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, states, actions, rewards, masks=None, attention_mask=None):
        # "masked" tokens or unspecified inputs can be passed in as None
        return None, None, None

    def get_action(self, states, actions, rewards, **kwargs):
        # these will come as tensors on the correct device
        return torch.zeros_like(actions[-1])

class DT(TrajT):
    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(self, environment_config, transformer_config, **kwargs):
        super().__init__(**kwargs)

        self.transformer_config = transformer_config
        self.environment_config = environment_config
        self.model_type = "decision_transformer"

        # state_dim, act_dim, hidden_size=128, max_length=None, max_ep_len=1001, action_tanh=False, **kwargs
        self.state_dim = np.prod(environment_config.observation_space["image"].shape)
        self.act_dim = environment_config.action_space.n
        self.hidden_size = transformer_config.d_model
        self.max_ep_len = environment_config.max_steps + 1

        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_positions=1024,
            n_embd=self.hidden_size,                    # d_model
            n_layer=transformer_config.n_layers,        # n_layers
            # d_head
            n_head=self.transformer_config.n_heads,     # n_heads
            n_inner=self.transformer_config.d_mlp,      # d_mlp
            # n_ctx
            activation_function=self.transformer_config.activation_fn,  # act_fn
            # gated_mlp
            # normalization_type
            # seed
            # device
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        # n_ctx is not included in GPT2Config in this transformers version
        config.n_ctx = self.transformer_config.n_ctx
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(self.max_ep_len, self.hidden_size)
        self.embed_return = torch.nn.Linear(1, self.hidden_size, bias=False)
        self.embed_state = torch.nn.Linear(self.state_dim, self.hidden_size, bias=False)
        self.embed_action = torch.nn.Embedding(self.act_dim + 1, self.hidden_size)

        # transformer_config.gated_mlp = None
        # self.embed_ln = nn.LayerNorm(self.hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(self.hidden_size, self.state_dim)
        self.predict_action = nn.Linear(self.hidden_size, self.act_dim)
        self.predict_return = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        '''
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
        '''

        # embed each modality with a different head
        # get_state_embedding()
        states = rearrange(
            states,
            "batch block height width channel -> (batch block) (channel height width)"
        )
        state_embeddings = self.embed_state(
            states.type(torch.float32).contiguous()
        )   # (batch * block_size, n_embd)
        state_embeddings = rearrange(
            state_embeddings,
            "(batch block) n_embd -> batch block n_embd",
            block=seq_length,
        )

        # action_embeddings = self.embed_action(actions) : (128,1,1) -> (128,1,1,128)
        # get_action_embedding() : (128,1,1) -> (128,1,128)
        block_size = actions.shape[1]
        if block_size == 0:
            return None  # no actions to embed
        actions = rearrange(
            actions, "batch block action -> (batch block) action"
        )
        # I don't see why we need this but we do? Maybe because of the sequential?
        action_embeddings = self.embed_action(actions).flatten(1)
        action_embeddings = rearrange(
            action_embeddings,
            "(batch block) n_embd -> batch block n_embd",
            block=block_size,
        )

        # returns_embeddings = self.embed_return(returns_to_go)
        # get_reward_embedding()
        block_size = returns_to_go.shape[1]
        rtgs = rearrange(returns_to_go, "batch block rtg -> (batch block) rtg")
        returns_embeddings = self.embed_return(rtgs.type(torch.float32))
        returns_embeddings = rearrange(
            returns_embeddings,
            "(batch block) n_embd -> batch block n_embd",
            block=block_size,
        )

        # time_embeddings = self.embed_timestep(timesteps)
        # get_time_embedding()
        assert (
                timesteps.max() <= self.environment_config.max_steps
        ), "timesteps must be less than max_timesteps"

        block_size = timesteps.shape[1]
        timesteps = rearrange(
            timesteps, "batch block time-> (batch block) time"
        )
        time_embeddings = self.embed_timestep(timesteps)
        if self.transformer_config.time_embedding_type != "linear":
            time_embeddings = time_embeddings.squeeze(-2)
        time_embeddings = rearrange(
            time_embeddings,
            "(batch block) n_embd -> batch block n_embd",
            block=block_size,
        )

        # get_token_embeddings()
        # time embeddings are treated similar to positional embeddings
        seq_length = time_embeddings.shape[1]
        state_embeddings = state_embeddings + time_embeddings
        # action_embeddings = action_embeddings + time_embeddings

        if action_embeddings is not None:
            if action_embeddings.shape[1] < seq_length:
                assert (
                    action_embeddings.shape[1] == seq_length - 1
                ), "Action embeddings must be one timestep less than state embeddings"
                action_embeddings = (
                    action_embeddings
                    + time_embeddings[:, : action_embeddings.shape[1]]
                )
                trajectory_length = seq_length * 3 - 1
            else:
                action_embeddings = action_embeddings + time_embeddings
                trajectory_length = seq_length * 3
        else:
            trajectory_length = 2  # one timestep, no action yet

        returns_embeddings = returns_embeddings + time_embeddings

        # create token embeddings
        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        token_embeddings = torch.zeros(
            (batch_size, trajectory_length, self.transformer_config.d_model),
            dtype=torch.float32,
            device=state_embeddings.device,
        )  # batches, blocksize, n_embd

        if action_embeddings is not None:
            token_embeddings[:, ::3, :] = returns_embeddings
            token_embeddings[:, 1::3, :] = state_embeddings
            token_embeddings[:, 2::3, :] = action_embeddings
        else:
            token_embeddings[:, 0, :] = returns_embeddings[:, 0, :]
            token_embeddings[:, 1, :] = state_embeddings[:, 0, :]

        '''
        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_size)
        # stacked_inputs = self.embed_ln(stacked_inputs)
        '''

        # make stacked_attention_mask with torch.ones since all inputs are used
        '''
        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3 * seq_length)
        '''
        stacked_attention_mask = torch.ones((batch_size, trajectory_length), dtype=torch.long).to(self.transformer_config.device)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=token_embeddings,
            attention_mask=stacked_attention_mask,
        )

        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        # x = x.reshape(batch_size, trajectory_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        if (x.shape[1] % 3 != 0) and ((x.shape[1] + 1) % 3 == 0):
            x = torch.concat((x, x[:, -2].unsqueeze(1)), dim=1)

        x = x.reshape(
            batch_size, seq_length, 3, self.transformer_config.d_model
        )
        x = x.permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_return(x[:, 2])  # predict next return given state and action
        state_preds = self.predict_state(x[:, 2])  # predict next state given state and action
        action_preds = self.predict_action(x[:, 1])  # predict next action given state

        return state_preds, action_preds, return_preds

    def get_action(self, states, actions, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            returns_to_go = returns_to_go[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length - states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length - states.shape[1], self.state_dim),
                             device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length - returns_to_go.shape[1], 1),
                             device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length - timesteps.shape[1]), device=timesteps.device),
                 timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, return_preds = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)

        return action_preds[0, -1]

    def get_action_with_context(self, states, actions, returns_to_go, timesteps, context, **kwargs):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            returns_to_go = returns_to_go[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat(
                [torch.zeros(self.max_length - states.shape[1]), torch.ones(states.shape[1])]
            )
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(
                1, -1
            )
            states = torch.cat(
                [
                    torch.zeros(
                        (states.shape[0], self.max_length - states.shape[1], self.state_dim),
                        device=states.device,
                    ),
                    states,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            actions = torch.cat(
                [
                    torch.zeros(
                        (actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                        device=actions.device,
                    ),
                    actions,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [
                    torch.zeros(
                        (returns_to_go.shape[0], self.max_length - returns_to_go.shape[1], 1),
                        device=returns_to_go.device,
                    ),
                    returns_to_go,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            timesteps = torch.cat(
                [
                    torch.zeros(
                        (timesteps.shape[0], self.max_length - timesteps.shape[1]),
                        device=timesteps.device,
                    ),
                    timesteps,
                ],
                dim=1,
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        ## add context to each input
        # states
        for i in range(20):
            states[0][i] = torch.from_numpy(context["observations"][i]).to(device=states.device, dtype=torch.float32)
        # actions
        for i in range(20):
            actions[0][i] = torch.from_numpy(context["actions"][i]).to(device=actions.device, dtype=torch.float32)
        # returns_to_go
        for i in range(20):
            returns_to_go[0][i] = context["returns_to_go"][i]
        # timesteps
        for i in range(20):
            timesteps[0][i] = context["timesteps"][i]
        # attention_mask
        for i in range(20):
            attention_mask[0][i] = 1

        _, action_preds, return_preds = self.forward(
            states,
            actions,
            None,
            returns_to_go,
            timesteps,
            attention_mask=attention_mask,
            **kwargs
        )

        return action_preds[0, -1]