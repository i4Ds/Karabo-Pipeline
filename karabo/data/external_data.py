import functools
import os
import re
import shutil
import warnings
from typing import Final, List

import requests
from tqdm import tqdm

from karabo.util._types import FilePathType
from karabo.util.file_handler import FileHandler

cscs_base_url = "https://object.cscs.ch/v1/AUTH_1e1ed97536cf4e8f9e214c7ca2700d62"
cscs_karabo_public_base_url = f"{cscs_base_url}/karabo_public"
cscs_karabo_public_testing_base_url = f"{cscs_karabo_public_base_url}/testing"


class DownloadObject:
    """Download handler for remote files & dirs.

    Important: There is a remote file-naming-convention, to be able to provide
    updates of cached dirs/files & to simplify maintainability.
    The convention for each object is <dirname><file|dir>_v<version>, where the version
    should be an integer, starting from 1. <dirname> should be the same as <file|dir>.
    The additional <dirname> is to have a single directory for each object, so that
    additional file/dir versions don't disturb the overall remote structure.

    The version of a downloaded object is determined by the current version of Karabo,
    meaning that they're hard-coded. Because Karabo relies partially on remote-objects,
    we don't guarantee their availability for deprecated Karabo versions.
    """

    URL_SEP: Final = "/"

    def __init__(
        self,
        remote_base_url: str,
    ) -> None:
        self.remote_base_url = remote_base_url

    @staticmethod
    def download(
        url: str,
        local_file_path: FilePathType,
        verify: bool = True,
        verbose: bool = True,
    ) -> int:
        download_dir = os.path.dirname(local_file_path)
        dir_existed = False
        if os.path.exists(download_dir):
            dir_existed = True
        try:
            response = requests.get(url, stream=True, verify=verify)
            if response.status_code != 200:
                response.raise_for_status()  # Will only raise for 4xx codes, so...
                raise RuntimeError(
                    f"Request to {url} returned status code {response.status_code}"
                )
            file_size = int(response.headers.get("Content-Length", 0))
            os.makedirs(download_dir, exist_ok=True)

            desc = f"Downloading {url} to {local_file_path}"
            response.raw.read = functools.partial(
                response.raw.read, decode_content=True
            )
            with tqdm.wrapattr(
                response.raw,
                "read",
                total=file_size,
                desc=desc,
                disable=not verbose,
            ) as r_raw, open(local_file_path, "wb") as f:
                shutil.copyfileobj(r_raw, f)

        except BaseException:  # cleanup if interrupted
            if os.path.exists(local_file_path):
                if os.path.isdir(local_file_path) and not dir_existed:
                    shutil.rmtree(local_file_path)
                else:
                    os.remove(local_file_path)
            raise
        return response.status_code

    def get_object(
        self,
        remote_file_path: str,
        verify: bool = True,
        verbose: bool = True,
    ) -> str:
        if verbose:
            purpose = "download-objects caching"
        else:
            purpose = None
        local_cache_dir = FileHandler().get_tmp_dir(
            prefix="objects-download-",
            term="long",
            purpose=purpose,
        )
        local_file_path = os.path.join(
            local_cache_dir,
            os.path.join(
                *remote_file_path.split(DownloadObject.URL_SEP)
            ),  # convert to local filesys.sep
        )
        if not os.path.exists(local_file_path):
            remote_url = (
                f"{self.remote_base_url}{DownloadObject.URL_SEP}{remote_file_path}"
            )
            _ = DownloadObject.download(
                url=remote_url,
                local_file_path=local_file_path,
                verify=verify,
                verbose=verbose,
            )
        return local_file_path

    @staticmethod
    def is_url_available(url: str) -> bool:
        """Checks whether the url is available or not.

        Returns:
            Ture if available, else False
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=Warning
            )  # filter `InsecureRequestWarning` from `urllib3`
            resp = requests.get(
                url=url,
                headers={"Range": "bytes=0-0"},
                verify=False,  # ok here, because no content is really fetched
            )
        return resp.status_code == 206


class SingleFileDownloadObject(DownloadObject):
    """Abstract single object download handler."""

    def __init__(
        self,
        remote_file_path: str,
        remote_base_url: str,
    ) -> None:
        self.remote_file_path = remote_file_path
        super().__init__(remote_base_url=remote_base_url)

    def get(
        self,
        verify: bool = True,
        verbose: bool = True,
    ) -> str:
        return super().get_object(
            remote_file_path=self.remote_file_path,
            verify=verify,
            verbose=verbose,
        )

    def is_available(self) -> bool:
        remote_url = (
            f"{self.remote_base_url}{DownloadObject.URL_SEP}{self.remote_file_path}"
        )
        return DownloadObject.is_url_available(url=remote_url)


class GLEAMSurveyDownloadObject(SingleFileDownloadObject):
    def __init__(self) -> None:
        super().__init__(
            remote_file_path="GLEAM_EGC.fits",
            remote_base_url=cscs_karabo_public_base_url,
        )


class HISourcesSmallCatalogDownloadObject(SingleFileDownloadObject):
    def __init__(self) -> None:
        super().__init__(
            remote_file_path="HI_sources_small_catalog.h5",
            remote_base_url=cscs_karabo_public_base_url,
        )


class BATTYESurveyDownloadObject(SingleFileDownloadObject):
    def __init__(self) -> None:
        raise NotImplementedError(
            f"Currently not available at {cscs_karabo_public_base_url}"
        )


class DilutedBATTYESurveyDownloadObject(SingleFileDownloadObject):
    def __init__(self) -> None:
        raise NotImplementedError(
            """This catalog has incorrect flux data.
            Use HISourcesSmallCatalogDownloadObject instead."""
        )


class MIGHTEESurveyDownloadObject(SingleFileDownloadObject):
    def __init__(self) -> None:
        super().__init__(
            remote_file_path="MIGHTEE_Continuum_Early_Science_COSMOS_Level1.fits",
            remote_base_url=cscs_karabo_public_base_url,
        )


class ExampleHDF5Map(SingleFileDownloadObject):
    def __init__(self) -> None:
        super().__init__(
            remote_file_path="example_map.h5",
            remote_base_url=cscs_karabo_public_base_url,
        )


class DiffuseEmissionHaslam408DownloadObject(SingleFileDownloadObject):
    def __init__(self) -> None:
        super().__init__(
            remote_file_path="lambda_mollweide_haslam408_dsds.fits",
            remote_base_url="https://lambda.gsfc.nasa.gov/data/foregrounds/haslam/images",  # noqa: E501
        )


class ContainerContents(DownloadObject):
    def __init__(
        self,
        remote_url: str,
        regexr_pattern: str,
    ) -> None:
        """
        Class for handling container contents downloaded from a URL.
        Also useful to see what is available in a container.

        Parameters
        ----------
        remote_base_url: str
            See `DownloadObject.remote_base_url`.
        regexr_pattern : str
            Regex pattern to match the desired contents in the directory.

        Examples
        --------
        >>> from karabo.data.external_data import (
        >>>     ContainerContents,
        >>>     cscs_karabo_public_base_url,
        >>> )
        >>> container_contents = ContainerContents(
        >>>     remote_base_url = cscs_karabo_public_base_url,
        >>>     regexr_pattern = "MGCLS/Abell_(?:2744)_.+_I_.+",
        >>> )
        >>> container_contents.get_file_paths()
        ["MGCLS/Abell_2744_aFix_pol_I_15arcsec_5pln_cor.fits.gz"]
        >>> download_object = DownloadObject("MGCLS/Abell_2744_aFix_pol_I_15arcsec_5pln_cor.fits.gz") # noqa
        """
        self.regexr_pattern = regexr_pattern
        self._remote_container_url = remote_url
        super().__init__(remote_base_url=remote_url)

    def get_container_content(self) -> str:
        """Gets the remote container-content as str."""
        response = requests.get(self._remote_container_url)
        response.raise_for_status()
        return response.text

    def get_file_paths(self) -> List[str]:
        """Applies `regexr_pattern` to container-objects."""
        xml_content = self.get_container_content()
        url_pattern = re.compile(self.regexr_pattern)
        urls = url_pattern.findall(xml_content)
        return urls

    def is_available(self) -> bool:
        """Checks if the container itself is available, not specific files.
        Is dependent on the `regexr_pattern`."""
        return len(self.get_file_paths()) > 0

    def get_all(self, verbose: bool = True) -> List[str]:
        """Gets all objects with the according cache paths as a list."""
        local_file_paths: List[str] = list()
        for remote_file_path in self.get_file_paths():
            local_file_path = self.get_object(
                remote_file_path=remote_file_path,
                verbose=verbose,
            )
            local_file_paths.append(local_file_path)
        return local_file_paths


class MGCLSContainerDownloadObject(ContainerContents):
    def __init__(
        self,
        regexr_pattern: str,
    ) -> None:
        super().__init__(
            remote_url=cscs_karabo_public_base_url,
            regexr_pattern=f"MGCLS{DownloadObject.URL_SEP}{regexr_pattern}",
        )
