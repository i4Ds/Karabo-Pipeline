import os
import re
import site
from typing import List, Optional

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
        try:
            response = requests.get(self.url, stream=True)
            response.raise_for_status()
            # Check that path folder exists
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "wb") as file:
                for chunk in response.iter_content(
                    chunk_size=8192
                ):  # Download in 8KB chunks
                    file.write(chunk)
        except BaseException:
            # Remove the file if the download is interrupted
            if os.path.exists(self.path):
                os.remove(self.path)
            raise

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


class FitsGzDownloadObject(DownloadObject):
    def __init__(self, file_name: str, folder_name: Optional[str]) -> None:
        base_path = (
            "https://object.cscs.ch/v1/AUTH_1e1ed97536cf4e8f9e214c7ca2700d62/"
            "karabo_public"
        )
        if folder_name is not None:
            base_path += f"/karabo_public/{folder_name}"
        super().__init__(
            file_name,
            f"{base_path}/{file_name}",
        )


class ContainerContents:
    def __init__(self, regexr_pattern: str):
        self.regexr_pattern = regexr_pattern

    def get_container_content(self) -> str:
        url = (
            "https://object.cscs.ch/v1/AUTH_1e1ed97536cf4e8f9e214c7ca2700d62"
            "/karabo_public"
        )
        # Download the XML content
        response = requests.get(url)

        # Make sure the request was successful
        response.raise_for_status()

        # Get the XML content as a string
        return response.text

    def get_file_paths(self) -> List[str]:
        xml_content = self.get_container_content()
        url_pattern = re.compile(self.regexr_pattern)
        urls = url_pattern.findall(xml_content)
        return urls


class MGCLSFilePaths(ContainerContents):
    def __init__(self, regexr_pattern: str) -> None:
        super().__init__(f"MGCLS/{regexr_pattern}")


class MGCLSFitsGzDownloadObject(FitsGzDownloadObject):
    def __init__(self, file_name: str) -> None:
        super().__init__(file_name, None)
