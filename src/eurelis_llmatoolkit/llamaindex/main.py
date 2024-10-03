from .config import ConfigLoader
from .indexation_wrapper import IndexationWrapper

# Pour les tests
if __name__ == "__main__":

    config = ConfigLoader.load_config("./eurelis_llmatoolkit/llamaindex/config.json")
    indexation_wrapper = IndexationWrapper(config)
    # indexation_wrapper.run_test()
    indexation_wrapper.run_indexation()
