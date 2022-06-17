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
            print(f"{self.name} is not downloaded yet. Downloading and caching for future uses...")
            self.__download()
        return self.path


class GLEAMSurveyDownloadObject(DownloadObject):

    def __init__(self):
        super().__init__("GLEAM_ECG.fits",
                         "https://swiss-ska.fhnw.ch/index.php/s/TFqpCeL882PDqKM/download/GLEAM_EGC.fits")


class ExampleHDF5Map(DownloadObject):
    
    def __init__(self):
        super().__init__("example_map.h5", "https://swiss-ska.fhnw.ch/index.php/s/Pnrm5bi2QPx9mNz/download/exmaple_map.h5")