from pytorch_lightning.overrides import LightningDistributedModule
from pytorch_lightning.strategies import DDPStrategy


class DDPStaticGraphStrategy(DDPStrategy):
    """Hack to fix DDP with gradient checkpointing"""
    def configure_ddp(self):
        self.pre_configure_ddp()
        self._model = self._setup_model(LightningDistributedModule(self.model))
        self._register_ddp_hooks()
        self._model._set_static_graph()
