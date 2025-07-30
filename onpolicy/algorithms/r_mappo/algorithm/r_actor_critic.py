import torch
import torch.nn as nn
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space

from onpolicy.algorithms.r_mappo.algorithm.sumo_nn import *
from onpolicy.algorithms.r_mappo.algorithm.improvements.time_enhancement import PositionalEncoding

from onpolicy.envs.sumo_files_marl.config import config

class R_Actor_SUMO(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor_SUMO, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        self.state_keys = config['environment']['state_key']
        self._policy_head = ActorModel(112, 8)
        self.to(device)

    def forward(self, obs, policy_head_input, masks, available_actions=None, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        logits = self._policy_head(policy_head_input)
        
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
            logits[available_actions==0] = -1e10
        
        action_logits = torch.distributions.Categorical(logits=logits)
        actions = logits.argmax(1) if deterministic else action_logits.sample()
        action_log_probs = action_logits.log_prob(actions)
        entropy = action_logits.entropy().unsqueeze(-1)
        return actions, action_log_probs

    def evaluate_actions(self, obs, action, policy_head_input, masks, available_actions=None, active_masks=None):
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        logits = self._policy_head(policy_head_input)
        
        if available_actions is not None:
            logits[available_actions==0] = -1e9
        
        action_logits = torch.distributions.Categorical(logits=logits)
        action_log_probs = action_logits.log_prob(action.squeeze(1)).unsqueeze(-1)
        
        if active_masks is not None:
            dist_entropy = (action_logits.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum()
        else:
            dist_entropy = action_logits.entropy().mean()

        return action_log_probs, dist_entropy

class R_Critic_SUMO(nn.Module):
    def __init__(self, hidden_size, device):
        super(R_Critic_SUMO, self).__init__()
        self.name = 'critic'
        self.tpdv = dict(dtype=torch.float32, device=device)
        self._value_head = CriticModel(112)
        self.to(device)

    def forward(self, policy_head_input):
        policy_head_input = check(policy_head_input).to(**self.tpdv)
        v, p = self._value_head(policy_head_input)
        return v, p

class shared_NN(nn.Module):
    def __init__(self, device):   
        super(shared_NN, self).__init__()
        self.name = 'shared_NN'
        self.iner_ps_encoding = PositionalEncoding(
            d_model=112,
            max_len=17,
            device=device
        )
        self.outer_ps_encoding = PositionalEncoding(
            d_model=112,
            max_len=25,
            device=device
        )
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.shared_NN = TransformerEncoderPolicy(self.iner_ps_encoding, self.outer_ps_encoding, device=device)
        self.to(device)

    def forward(self, observations):
        observations = check(observations).to(**self.tpdv)
        policy_head_input, transformer_output, pred, pred_ele, pred_mask = self.shared_NN.compute_memories(observations)
        return policy_head_input, pred, pred_ele, pred_mask

class R_Actor(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)
        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features, action, available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None
        )
        return action_log_probs, dist_entropy

class R_Critic(nn.Module):
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(args, cent_obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        if self._use_popart:
            self.v_out = init(PopArt(self.hidden_size, 1, device=device), init_method, lambda x: nn.init.constant_(x, 0))
        else:
            self.v_out = init(nn.Linear(self.hidden_size, 1), init_method, lambda x: nn.init.constant_(x, 0))
        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)
        return values, rnn_states
