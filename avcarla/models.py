from avstack.config import MODELS, ConfigDict
from avstack.modules import BaseModule
from avstack.utils.decorators import apply_hooks


@MODELS.register_module()
class CarlaImageDetector(BaseModule):
    def __init__(self, algorithm: ConfigDict, *args, **kwargs):
        super().__init__(name="CarlaImageDetector", *args, **kwargs)
        self.algorithm = MODELS.build(algorithm)

    @apply_hooks
    def __call__(self, image):
        # output = self.algorithm(image)
        return None

    def initialize(self, *args, **kwargs):
        pass
