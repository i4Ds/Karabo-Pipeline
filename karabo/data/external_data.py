import os
import site

import requests


class KaraboCache:
    base_path: str = site.getsitepackages()[0]
    use_scratch_folder_if_exist: bool = True

    if "SCRATCH" in os.environ and use_scratch_folder_if_exist:
        base_path = os.environ["SCRATCH"]

    @staticmethod
    def valida_cache_directory_exists() -> None:
        cache_path = KaraboCache.get_cache_directory()
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)

    @staticmethod
    def get_cache_directory() -> str:
        cache_path = f"{KaraboCache.base_path}/karabo_cache"
        return cache_path


class DownloadObject:
    def __init__(
        self,
        name: str,
        url: str,
    ) -> None:
        self.name = name
        self.url = url
        KaraboCache.valida_cache_directory_exists()
        directory = KaraboCache.get_cache_directory()
        self.path = f"{directory}/{name}"

    def __download(self) -> None:
        response = requests.get(self.url, stream=True)
        response.raise_for_status()
        with open(self.path, "wb") as file:
            for chunk in response.iter_content(
                chunk_size=8192
            ):  # Download in 8KB chunks
                file.write(chunk)

    def __is_downloaded(self) -> bool:
        if os.path.exists(self.path):
            return True
        return False

    def get(self) -> str:
        if not self.__is_downloaded():
            print(f"{self.name} is not downloaded yet.")
            print("Downloading and caching for future uses to " f"{self.path} ...")
            self.__download()
        return self.path

    def is_available(self) -> bool:
        """Checks whether the url is available or not.

        Returns:
            Ture if available, else False
        """
        resp = requests.get(
            url=self.url,
            headers={"Range": "bytes=0-0"},
        )
        if resp.status_code == 206:  # succeed & partial content
            return True
        else:
            return False


class GLEAMSurveyDownloadObject(DownloadObject):
    def __init__(self) -> None:
        super().__init__(
            "GLEAM_ECG.fits",
            "https://object.cscs.ch/v1/AUTH_1e1ed97536cf4e8f9e214c7ca2700d62"
            + "/karabo_public/GLEAM_EGC.fits",
        )


class BATTYESurveyDownloadObject(DownloadObject):
    def __init__(self) -> None:
        super().__init__(
            "point_sources_OSKAR1_battye.h5",
            "https://object.cscs.ch/v1/AUTH_1e1ed97536cf4e8f9e214c7ca2700d62"
            + "/karabo_public/point_sources_OSKAR1_battye.h5",
        )


class DilutedBATTYESurveyDownloadObject(DownloadObject):
    def __init__(self) -> None:
        super().__init__(
            "point_sources_OSKAR1_diluted5000.h5",
            "https://object.cscs.ch/v1/AUTH_1e1ed97536cf4e8f9e214c7ca2700d62"
            + "/karabo_public/point_sources_OSKAR1_diluted5000.h5",
        )


class MIGHTEESurveyDownloadObject(DownloadObject):
    def __init__(self) -> None:
        super().__init__(
            "MIGHTEE_Continuum_Early_Science_COSMOS_Level1.fits",
            "https://object.cscs.ch:443/v1/AUTH_1e1ed97536cf4e8f9e214c7ca2700d62"
            + "/karabo_public/MIGHTEE_Continuum_Early_Science_COSMOS_Level1.fits",
        )


class ExampleHDF5Map(DownloadObject):
    def __init__(self) -> None:
        super().__init__(
            "example_map.h5",
            "https://object.cscs.ch/v1/AUTH_1e1ed97536cf4e8f9e214c7ca2700d62"
            + "/karabo_public/example_map.h5",
        )
