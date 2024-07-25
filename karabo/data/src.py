from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union, cast
from warnings import warn

from karabo.data.obscore import ObsCoreMeta
from karabo.util._types import FilePathType, TFilePathType


@dataclass
class RucioMeta:
    """Metadata dataclass to handle SKA SRC Ingestion for Rucio.

    This dataclass may go through some changes in the future in case
        the Rucio service is also changing.

    See `https://gitlab.com/ska-telescope/src/ska-src-ingestion/-/tree/main`.

    Args:
        namespace: The Rucio scope in which the new file should be located.

        name: The name of the file within Rucio - the scope:name together is the
            Data Identifier (DID).

        lifetime: The lifetime in seconds for the new file to be retained.

        dataset_name: The Rucio dataset name the file will be attached to.
            The dataset scope will be the same as that specified in namespace.

        meta: An object containing science metadata fields, which will be set against
            the ingested file. This should be either a dict of `ObsCoreMeta` or an
            instance of `ObsCoreMeta`.
    """

    namespace: str
    name: str
    lifetime: int
    dataset_name: Optional[str] = None
    meta: Optional[Union[Dict[str, Any], ObsCoreMeta]] = None

    def to_dict(
        self,
        fpath: Optional[FilePathType] = None,
        *,
        ignore_none: bool = True,
    ) -> Dict[str, Any]:
        """Converts this dataclass into a dict.

        Args:
            fpath: File-path to write dump. Consider using `get_meta_fname`
                to get an `fpath` according to the Rucio specification.
            ignore_none: Ignore `None` fields?

        Returns:
            Dataclass as dict.
        """
        if self.meta is not None and isinstance(self.meta, ObsCoreMeta):
            self_new = deepcopy(self)  # to avoid mutable `self.to_json`
            self_new.meta = self.meta.to_dict(fpath=None, ignore_none=ignore_none)
        else:
            self_new = self
        dictionary = asdict(self_new)
        if ignore_none:
            dictionary = {
                key: value for key, value in dictionary.items() if value is not None
            }
        if fpath is not None:
            meta_suffix = RucioMeta._metadata_suffix()
            if not str(fpath).endswith(meta_suffix):
                wmsg = f"Provided {fpath=} doesn't end with {meta_suffix=}"
                warn(message=wmsg, category=UserWarning, stacklevel=1)
            dump = json.dumps(dictionary)
            with open(file=fpath, mode="w") as json_file:
                json_file.write(dump)
        return dictionary

    @classmethod
    def get_meta_fname(cls, fname: TFilePathType) -> TFilePathType:
        """Gets the metadata-filename of `fname`.

        It's according to the Rucio metadata specification (if up-to-date). The
            specification states that metadata is expected to be provided by two files:
            fname: `data_name` and metadata: `data_name.<metadata_suffix>`, where the
            suffix is set to `meta`.

        Args:
            fname: Filename to create metadata filename from.

        Returns:
            Metadata filename (or filepath if `fname` was also a filepath).
        """
        meta_suffix = cls._metadata_suffix()
        meta_fname = f"{fname}.{meta_suffix}"
        if isinstance(fname, str):
            return cast(TFilePathType, meta_fname)
        elif isinstance(fname, Path):
            return cast(TFilePathType, Path(meta_fname))
        else:
            err_msg = f"Unexpected {type(fname)=} of {fname=}."
            raise TypeError(err_msg)  # `assert_never`` doesn't work here

    @classmethod
    def get_ivoid(
        cls,
        *,
        authority: str = "test.skao",
        path: str = "/~",
        namespace: str,
        name: str,
        fragment: Optional[str] = None,
    ) -> str:
        """Gets the IVOA identifier for `ObsCoreMeta.obs_creator_did`.

        SRCNet Rucio IVOID according to IVOA 'REC-Identifiers-2.0'. Do NOT specify
            RFC 3986 delimiters in the input-args, they're added automatically.

        Please set up an Issue if this is not up-to-date anymore.

        Args:
            authority: Organization (usually a data provider) that has been granted
                the right by the IVOA to create IVOA-compliant identifiers for
                resources it registers.
            path: Resource key. It's 'a resource that is unique within the namespace
                of an authority identifier.
            namespace: `RucioMeta.namespace`.
            name: `RucioMeta.name` (filename in Rucio).
            fragment: According to RFC 3986.

        Returns:
            IVOID.
        """
        query = f"{namespace}:{name}"  # according to current [07/2024] implementation
        return ObsCoreMeta.get_ivoid(
            authority=authority,
            path=path,
            query=query,
            fragment=fragment,
        )

    @classmethod
    def _metadata_suffix(cls) -> str:
        """Gets metadata suffix.

        Returns:
            Metadata suffix.
        """
        return "meta"
