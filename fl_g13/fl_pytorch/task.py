from collections import OrderedDict
import torch
from flwr.common import RecordDict

def get_weights(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_weights(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def _compute_task_vector(updated_weights, pre_trained_weights, client_state: RecordDict):
        """compute τ = (θ* − θ₀) ⊙ mask"""
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fine_tuned_weights_tensors = [torch.tensor(w, device=dev) for w in updated_weights]
        pre_trained_weights_tensors = [torch.tensor(w, device=dev) for w in pre_trained_weights]
        #compressed_mask_list = self.client_state.config_records['mask']['mask_list']
        #mask_list = uncompress_mask_sparse(compressed_mask_list, device=self.device)
        mask_list = client_state.array_records['mask'].to_numpy_ndarrays()
        mask_list = [torch.tensor(mask, device=dev) for mask in mask_list]
        task_vector = [
            mask_layer * (fine_tuned_layer - pre_trained_layer)
            for fine_tuned_layer, pre_trained_layer, mask_layer in zip(
                fine_tuned_weights_tensors, 
                pre_trained_weights_tensors, 
                mask_list
            )
        ]
        # Convert to type required by Flower
        fit_params = [layer.cpu().numpy() for layer in task_vector]
        return fit_params