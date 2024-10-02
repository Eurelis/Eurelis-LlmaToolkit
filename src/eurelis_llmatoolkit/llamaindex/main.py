from indexation_wrapper import IndexationWrapper

# Pour les tests
if __name__ == "__main__":
    from config import ConfigLoader

    config = ConfigLoader.load_config("config.json")
    indexation_wrapper = IndexationWrapper(config)
    # indexation_wrapper.run_test()
    indexation_wrapper.run_indexation()