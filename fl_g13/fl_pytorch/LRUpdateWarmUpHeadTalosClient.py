from fl_g13.fl_pytorch.warm_up_head_talos_client import WarmUpHeadTalosClient

class LRUpdateWarmUpHeadTalosClient(WarmUpHeadTalosClient):

    def fit(self, parameters, config):

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = config['lr']
            print(f"Updated learning rate to {param_group['lr']}")

        return super().fit(parameters, config)