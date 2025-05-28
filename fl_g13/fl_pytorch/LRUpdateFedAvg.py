import math
from fl_g13.fl_pytorch.strategy import CustomFedAvg


WANDB_PROJECT_NAME = "CIFAR100_FL_experiment"


# *** -------- AGGREGATION SERVER STRATEGY -------- *** #

class LRUpdateFedAvg(CustomFedAvg):
    def __init__(
        self,
        checkpoint_dir,
        prefix,
        model,
        start_epoch=1,
        save_every=1,
        save_with_model_dir=False,
        use_wandb = False,
        wandb_config=None,
        sparsity=0.8,
        num_warmup_rounds=1,
        initial_lr = 1e-3,
        eta_min = 1e-5,
        T_max = 120,
        *args,
        **kwargs
    ):
        super().__init__(
            checkpoint_dir=checkpoint_dir,
            prefix=prefix,
            model=model,
            start_epoch=start_epoch,
            save_every=save_every,
            save_with_model_dir=save_with_model_dir,
            use_wandb=use_wandb,
            wandb_config=wandb_config,
            *args,
            **kwargs
        )

        self.initial_lr = initial_lr
        self.eta_min = eta_min
        self.T_max = T_max

    # -------- AGGREGATION -------- #

    def configure_fit(self, server_round, parameters, client_manager):
        """
        Pass the current learning rate to the clients using a CosineAnnealing schedule.
        """
        print("Entered server configure_fit()")
        # CosineAnnealingLR: eta_min + (initial_lr - eta_min) * (1 + cos(pi * T_cur / T_max)) / 2
        # We'll assume initial_lr and eta_min are attributes or can be set as defaults.
        # T_max is the total number of rounds (or a reasonable guess).

        # Compute the new learning rate for this round
        T_cur = server_round - 1  # 0-based
        if T_cur > self.T_max:
            T_cur = self.T_max
        lr = self.eta_min + (self.initial_lr - self.eta_min) * (1 + math.cos(math.pi * T_cur / self.T_max)) / 2

        # Call the parent to get the client instructions
        client_instructions = super().configure_fit(server_round, parameters, client_manager)

        # Add the learning rate to each client's config
        for client_id, fit_ins in client_instructions:
            fit_ins.config["lr"] = lr

        return client_instructions