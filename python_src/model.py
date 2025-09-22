import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- CONSTANTS ---
NUM_FEATURE_CHANNELS = 16
NUM_SUITS = 4
NUM_RANKS = 13
ACTION_VECTOR_SIZE = 208

class ResBlock(nn.Module):
    """
    Residual block with GroupNorm for stable feature extraction.
    """
    def __init__(self, channels, groups=8):
        super().__init__()
        self.gn1 = nn.GroupNorm(groups, channels)
        self.gn2 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.act = nn.SiLU(inplace=True)

        # Initialize the last layer to zero to make the block an identity function at the start
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x):
        identity = x
        out = self.act(self.gn1(x))
        out = self.conv1(out)
        out = self.act(self.gn2(out))
        out = self.conv2(out)
        out = out + identity # Key residual connection
        return out

class OFC_CNN_Network(nn.Module):
    """
    Optimized network architecture with a shared body and separate heads.
    """
    def __init__(self, hidden_size=512, channels=64, num_res_blocks=6, groups=8):
        super(OFC_CNN_Network, self).__init__()
        
        # --- Network "Body" (heavy part) ---
        self.body = nn.Sequential(
            nn.Conv2d(NUM_FEATURE_CHANNELS, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            *[ResBlock(channels, groups) for _ in range(num_res_blocks)],
        )
        
        conv_output_size = channels * NUM_SUITS * NUM_RANKS
        
        # --- Shared Fully Connected Part ---
        self.body_fc = nn.Sequential(
            nn.Linear(conv_output_size, hidden_size),
            nn.SiLU(inplace=True),
        )

        # --- Network "Heads" (lightweight parts) ---
        # 1. Value Head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # 2. Policy Head
        self.action_proj = nn.Sequential(
            nn.Linear(ACTION_VECTOR_SIZE, 128),
            nn.SiLU(inplace=True),
        )
        self.street_proj = nn.Sequential(
            nn.Linear(5, 8),
            nn.SiLU(inplace=True),
        )
        
        self.body_ln = nn.LayerNorm(hidden_size)
        self.action_ln = nn.LayerNorm(128)
        self.street_ln = nn.LayerNorm(8)
        
        policy_input_size = hidden_size + 128 + 8
        
        self.policy_head_fc = nn.Sequential(
            nn.Linear(policy_input_size, hidden_size),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_size // 2, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(m.bias, -bound, bound)

    def forward_body(self, state_image):
        conv_out = self.body(state_image)
        flat_out = conv_out.view(conv_out.size(0), -1)
        body_out = self.body_fc(flat_out)
        return body_out

    def forward_value_head(self, body_out):
        return self.value_head(body_out)

    def forward_policy_head(self, body_out, action_vec, street_vector):
        action_embedding = self.action_proj(action_vec)
        street_embedding = self.street_proj(street_vector)
        
        body_norm = self.body_ln(body_out)
        action_norm = self.action_ln(action_embedding)
        street_norm = self.street_ln(street_embedding)
        
        policy_input = torch.cat([body_norm, action_norm, street_norm], dim=1)
        policy_logit = self.policy_head_fc(policy_input)
        return policy_logit

    def forward(self, state_image, action_vec=None, street_vector=None):
        body_out = self.forward_body(state_image)
        
        if action_vec is None:
            # Value only
            value = self.forward_value_head(body_out)
            return value
        
        # Policy
        if street_vector is None:
            raise ValueError("street_vector must be provided for policy evaluation")
            
        policy_logit = self.forward_policy_head(body_out, action_vec, street_vector)
        
        return policy_logit, None
