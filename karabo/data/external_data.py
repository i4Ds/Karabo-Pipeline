import os
import site

import requests


class KaraboCache:
    @staticmethod
    def valida_cache_directory_exists():
        path = site.getsitepackages()[0]
        cache_path = f"{path}/karabo_cache"
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)

    @staticmethod
    def get_cache_directory():
        path = site.getsitepackages()[0]
        cache_path = f"{path}/karabo_cache"
        return cache_path


class DownloadObject:
    def __init__(self, name, url):
        self.name = name
        self.url = url
        KaraboCache.valida_cache_directory_exists()
        directory = KaraboCache.get_cache_directory()
        self.path = f"{directory}/{name}"

    def __download(self):
        response = requests.get(self.url)
        open(self.path, "wb").write(response.content)

    def __is_downloaded(self):
        if os.path.exists(self.path):
            return True
        return False

    def get(self) -> str:
        if not self.__is_downloaded():
            print(
                f"{self.name} is not downloaded yet. "
                + "Downloading and caching for future uses..."
            )
            self.__download()
        return self.path


class GLEAMSurveyDownloadObject(DownloadObject):
    def __init__(self):
        super().__init__(
            "GLEAM_ECG.fits",
            "https://object.cscs.ch/v1/AUTH_1e1ed97536cf4e8f9e214c7ca2700d62"
            + "/karabo_public/GLEAM_EGC.fits",
        )


class MIGHTEESurveyDownloadObject(DownloadObject):
    def __init__(self):
        super().__init__(
            "MIGHTEE_Continuum_Early_Science_COSMOS_Level1.fits",
            "https://object.cscs.ch:443/v1/AUTH_1e1ed97536cf4e8f9e214c7ca2700d62"
            + "/karabo_public/MIGHTEE_Continuum_Early_Science_COSMOS_Level1.fits",
        )


class ExampleHDF5Map(DownloadObject):
    def __init__(self):
        super().__init__(
            "example_map.h5",
            "https://object.cscs.ch/v1/AUTH_1e1ed97536cf4e8f9e214c7ca2700d62"
            + "/karabo_public/example_map.h5",
        )
