from eurelis_llmatoolkit.api.misc.base_config import BaseConfig

base_config = BaseConfig()


class VersionInfo:
    _branch_name = "No GIT Info"
    _commit_id = "No GIT Info"
    _commit_number = "No GIT Info"
    _commit_date = "No GIT Info"
    _tag_name = "No GIT Info"
    try:
        with open("VERSION_TAG") as f:
            content = f.readline().split("::")
            _branch_name = content[0]
            _commit_id = content[1]
            _commit_number = content[2]
            _commit_date = content[3]
            _tag_name = content[4]
    except Exception:
        pass

    @classmethod
    def get_version_info_string(cls) -> str:
        return f"{base_config.get_current_environment()} : {cls._branch_name} - {cls._commit_id} - {cls._commit_number} - {cls._commit_date}"

    @classmethod
    def get_version_info_dict(cls) -> dict:
        return {
            "current_environment": base_config.get_current_environment(),
            "branch_name": cls._branch_name,
            "commit_id": cls._commit_id,
            "commit_number": cls._commit_number,
            "commit_date": cls._commit_date,
            "tag_name": cls._tag_name,
        }
