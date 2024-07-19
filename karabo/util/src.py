from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

from typing_extensions import assert_never

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
            the ingested file. Currently, this should be either a JSON or `ObsCoreMeta`.
    """

    namespace: str
    name: str
    lifetime: int
    dataset_name: Optional[str] = None
    meta: Optional[Union[str, ObsCoreMeta]] = None

    def to_json(
        self,
        fpath: Optional[FilePathType] = None,
        ignore_none: bool = True,
    ) -> str:
        """Converts this dataclass into a JSON.

        Args:
            fpath: JSON file-path to write dump. Consider using `get_meta_fname`
                to get and `fpath` according to the Rucio specification.
            ignore_none: Ignore `None` fields?

        Returns:
            JSON as a str.
        """
        self_copy = deepcopy(self)
        if self_copy.meta is not None and isinstance(self_copy.meta, ObsCoreMeta):
            self_copy.meta = self_copy.meta.to_json(fpath=None, ignore_none=ignore_none)
        dictionary = asdict(self_copy)
        if ignore_none:
            dictionary = {
                key: value for key, value in dictionary.items() if value is not None
            }
        dump = json.dumps(dictionary)
        if fpath is not None:
            with open(file=fpath, mode="w") as json_file:
                json_file.write(dump)
        return dump

    @classmethod
    def get_meta_fname(cls, fname: TFilePathType) -> TFilePathType:
        """Gets the metadata-filename of `fname`.

        It's according to the Rucio metadata specification (if up-to-date).

        Args:
            fname: Filename to create metadata filename from.

        Returns:
            Metadat filename (or filepath if `fname` was also a filepath).
        """
        meta_fname = f"{fname}.meta"
        if isinstance(fname, str):
            return meta_fname
        elif isinstance(fname, Path):
            return Path(meta_fname)
        else:
            assert_never(fname)


class RucioHandler:
    """SRCNet Rucio utils."""

    def __init__(self) -> None:
        """Constructor of `RucioHandler`."""
        super().__init__()
