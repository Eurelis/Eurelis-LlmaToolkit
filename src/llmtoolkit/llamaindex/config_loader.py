import json
import os
import re


class ConfigLoader:
    @staticmethod
    def load_config(config_path):
        with open(config_path, "r") as file:
            config = json.load(file)
        return ConfigLoader._replace_env_variables(config)

    @staticmethod
    def _replace_env_var_in_string(value):
        # Fonction qui remplace les variables d'environnement dans une chaîne de texte
        env_var_pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

        def replace(match):
            env_var = match.group(1)
            return os.getenv(
                env_var, match.group(0)
            )  # Remplace ou laisse ${VAR} tel quel

        return env_var_pattern.sub(replace, value)

    @staticmethod
    def _replace_env_variables(config):
        if isinstance(config, dict):
            for key, value in config.items():
                config[key] = ConfigLoader._replace_env_variables(value)
        elif isinstance(config, list):
            return [ConfigLoader._replace_env_variables(item) for item in config]
        elif isinstance(config, str):
            return ConfigLoader._replace_env_var_in_string(config)
        return config
