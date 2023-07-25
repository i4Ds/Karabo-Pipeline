import os
import re
import site
from typing import List

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
        cache_path = os.path.join(KaraboCache.base_path, "karabo_cache")
        return cache_path


class DownloadObject:
    def __init__(
        self,
        file_name: str,  # e.g. "karabo_public/point_sources_OSKAR1_battye.h5"
        base_url: str = "https://object.cscs.ch/v1/"
        + "AUTH_1e1ed97536cf4e8f9e214c7ca2700d62",
        container_name: str = "karabo_public",
    ) -> None:
        KaraboCache.valida_cache_directory_exists()
        directory = KaraboCache.get_cache_directory()
        # Create final paths
        self.local_path = os.path.join(directory, container_name, file_name)
        self.url = f"{base_url}/{container_name}/{file_name}"

    def __download(self) -> None:
        try:
            response = requests.get(self.url, stream=True)
            response.raise_for_status()
            # Check that path folder exists
            os.makedirs(os.path.dirname(self.local_path), exist_ok=True)
            with open(self.local_path, "wb") as file:
                for chunk in response.iter_content(
                    chunk_size=8192
                ):  # Download in 8KB chunks
                    file.write(chunk)
        except BaseException:
            # Remove the file if the download is interrupted
            if os.path.exists(self.local_path):
                os.remove(self.local_path)
            raise

    def __is_downloaded(self) -> bool:
        return os.path.exists(self.local_path)

    def get(self) -> str:
        if not self.__is_downloaded():
            print(f"{self.local_path.split(os.sep)[-1]} is not downloaded yet.")
            print("Downloading and caching for future uses to " f"{self.local_path} ..")
            self.__download()
        return self.local_path

    def is_available(self) -> bool:
        """Checks whether the url is available or not.

        Returns:
            Ture if available, else False
        """
        resp = requests.get(
            url=self.url,
            headers={"Range": "bytes=0-0"},
        )
        return resp.status_code == 206


class GLEAMSurveyDownloadObject(DownloadObject):
    def __init__(self) -> None:
        super().__init__(
            "GLEAM_EGC.fits",
        )


class BATTYESurveyDownloadObject(DownloadObject):
    def __init__(self) -> None:
        super().__init__(
            "point_sources_OSKAR1_battye.h5",
        )


class DilutedBATTYESurveyDownloadObject(DownloadObject):
    def __init__(self) -> None:
        super().__init__(
            "point_sources_OSKAR1_diluted5000.h5",
        )


class MIGHTEESurveyDownloadObject(DownloadObject):
    def __init__(self) -> None:
        super().__init__(
            "MIGHTEE_Continuum_Early_Science_COSMOS_Level1.fits",
        )


class ExampleHDF5Map(DownloadObject):
    def __init__(self) -> None:
        super().__init__(
            "example_map.h5",
        )


class ContainerContents:
    def __init__(self, regexr_pattern: str):
        """
        Class for handling container contents downloaded from a URL.
        Also useful to see what is available in a container.

        Parameters
        ----------
        regexr_pattern : str
            Regular expression pattern to match the desired contents in the container.

        Examples
        --------
        >>> from karabo.data.external_data import ContainerContents
        >>> container_contents = ContainerContents("MGCLS/Abell_(?:2744)_.+_I_.+")
        >>> container_contents.get_file_paths()
        ["MGCLS/Abell_2744_aFix_pol_I_15arcsec_5pln_cor.fits.gz"]
        >>> download_object = DownloadObject("MGCLS/Abell_2744_aFix_pol_I_15arcsec_5pln_cor.fits.gz") # noqa
        """
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


class MGCLSFitsGzDownloadObject(DownloadObject):
    def __init__(self, file_name: str) -> None:
        super().__init__(file_name)
