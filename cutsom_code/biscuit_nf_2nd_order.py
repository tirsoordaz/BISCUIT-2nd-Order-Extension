import torch
import torch.nn as nn

# Import the original baseline model so we can inherit from it
from models.biscuit_nf.lightning_module import BISCUITNF

# Import the necessary prior components
from models.shared.transition_prior import create_interaction_prior
from models.shared.modules import MultivarLinear, MultivarSequential

class BISCUITNF_2ndOrder(BISCUITNF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize the baseline prior normally
        self.prior_t1 = create_interaction_prior(num_latents=self.hparams.num_latents,
                                                 c_hid=self.hparams.c_hid,
                                                 add_prev_state=self.hparams.prior_action_add_prev_state,
                                                 action_size=self.hparams.action_size,
                                                 img_width=self.hparams.img_width,
                                                 extra_args=kwargs)

        # -----------------------------------------------------------------------------
        # MODIFICATION 1: Overwrite input layers for 2nd-Order History
        # -----------------------------------------------------------------------------
        
        # In the baseline (1st-order), the input layer expects a single previous state (size 1).
        # We change this to 2 to accept both t-2 and t-1.
        self.prior_t1.inp_layer = MultivarLinear(2, self.hparams.c_hid, [self.hparams.num_latents])
        
        # The context layer normally expects a condition vector of size `num_latents` (e.g., 32).
        # Because we will concatenate two time steps, we must double the input size to `num_latents * 2` (64).
        # We recreate the entire MultivarSequential block because it doesn't support index assignment.
        self.prior_t1.context_layer = MultivarSequential(
            MultivarLinear(self.hparams.num_latents * 2, self.hparams.c_hid, [self.hparams.num_latents]),
            nn.Tanh(),
            MultivarLinear(self.hparams.c_hid, self.hparams.c_hid, [self.hparams.num_latents])
        )

    def _get_loss(self, batch, mode='train'):
        """ Overridden to support 2nd-order temporal dependencies """
        if len(batch) == 2:
            x_enc, action = batch
        else:
            x_enc, _, action = batch
            
        with torch.no_grad():
            x_enc = x_enc[...,None,:].expand(-1, -1, self.hparams.num_samples, -1)
            batch_size, seq_len, num_samples, num_latents = x_enc.shape
            x_sample = x_enc + torch.randn_like(x_enc) * self.hparams.noise_level
            x_sample = x_sample.flatten(0, 2)
            
        z_sample, ldj = self.flow(x_sample)
        z_sample = z_sample.unflatten(0, (batch_size, seq_len, num_samples))
        ldj = ldj.reshape(batch_size, seq_len, num_samples)
        
        # -----------------------------------------------------------------------------
        # MODIFICATION 2: Extracting 3 timesteps instead of 2
        # -----------------------------------------------------------------------------
        # The baseline dynamically splits the sequence in half. 
        # Here, we explicitly grab our sequence of 3 (t-2, t-1, t).
        z_t_minus_2 = z_sample[:, 0]  # Shape: [batch_size, num_samples, num_latents]
        z_t_minus_1 = z_sample[:, 1]  # Shape: [batch_size, num_samples, num_latents]
        z_t         = z_sample[:, 2]  # Shape: [batch_size, num_samples, num_latents]
        
        # -----------------------------------------------------------------------------
        # MODIFICATION 3: Concatenating the history
        # -----------------------------------------------------------------------------
        # We attach z_t-2 and z_t-1 along the feature dimension (dim=-1).
        # If num_latents is 32, z_history now has 64 features.
        z_history = torch.cat([z_t_minus_2, z_t_minus_1], dim=-1)
        
        # -----------------------------------------------------------------------------
        # MODIFICATION 4: Correct Action Indexing & NLL Calculation
        # -----------------------------------------------------------------------------
        # Since our sequence length is 3, there are 2 actions. 
        # The action causing the transition from t-1 into t is at index 1.
        current_action = action[:, 1]
        
        # We pass our new concatenated z_history to the prior.
        # We removed the `.flatten(0, 1)` from the baseline because we already 
        # explicitly sliced out the sequence/time dimension above.
        nll = self.prior_t1.sample_based_nll(z_t=z_history,
                                             z_t1=z_t,
                                             action=current_action)
                                             
        # LDJ (Log Determinant Jacobian) is calculated for the target step (index 2)
        ldj = ldj[:, 2].mean(dim=-1) 
        
        loss = (nll + ldj).mean()

        self.log(f'{mode}_nll', nll.mean())
        self.log(f'{mode}_ldj', ldj.mean())

        return loss
