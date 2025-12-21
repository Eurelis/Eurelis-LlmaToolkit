class CacheFactory:
    @staticmethod
    def create_cache(cache_config: dict):
        provider = cache_config["provider"]

        if provider == "FSCache":
            from eurelis_llmatoolkit.llamaindex.scraping_cache.cache_marshaller.fs_cache_marshaller import (
                FSCacheMarshaller,
            )

            return FSCacheMarshaller(cache_config)

        raise ValueError(f"Cache provider {provider} is not supported.")
