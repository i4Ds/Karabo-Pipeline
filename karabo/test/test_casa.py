from __future__ import annotations

from karabo.data.casa import (
    MSAntennaTable,
    MSFieldTable,
    MSMainTable,
    MSMeta,
    MSObservationTable,
    MSPolarizationTable,
    MSSpectralWindowTable,
)
from karabo.simulation.visibility import Visibility


class TestCasaMS:
    def test_tables(self, minimal_casa_ms: Visibility) -> None:
        """Minimal table-creation test.

        This test currently just calls the table creation function of the particular
            to ensure field-name correctness & data loading success.

        Args:
            minimal_casa_ms: Casa MS fixture.
        """
        assert minimal_casa_ms.format == "MS"
        ms_path = minimal_casa_ms.path
        _ = MSAntennaTable.from_ms(ms_path=ms_path)
        _ = MSFieldTable.from_ms(ms_path=ms_path)
        _ = MSMainTable.from_ms(ms_path=ms_path)
        assert MSMainTable.ms_version(ms_path=ms_path) == "2.0"
        _ = MSObservationTable.from_ms(ms_path=ms_path)
        _ = MSPolarizationTable.from_ms(ms_path=ms_path)
        _ = MSSpectralWindowTable.from_ms(ms_path=ms_path)
        ms_meta = MSMeta.from_ms(ms_path=ms_path)
        assert ms_meta.ms_version == "2.0"
