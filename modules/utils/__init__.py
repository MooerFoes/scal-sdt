class TransformersNoStupidWarnings:
    """Silent warnings like "Some weights of the model checkpoint at openai/clip-vit-large-patch14 were not used"."""
    from transformers import logging

    def __enter__(self):
        self._prev_verbosity = self.logging.get_verbosity()
        self.logging.set_verbosity_error()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logging.set_verbosity(self._prev_verbosity)
