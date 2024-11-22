import enum

# when adding new versions make sure the string for the enum value
# is the string used in the telescope file name


class ALMAVersions(enum.Enum):
    """
    ALMA Cycle 0 configuration : {link}
    """

    ALL = "all"
    CYCLE_0_COMPACT = "cycle0.compact"
    CYCLE_0_EXTENDED = "cycle0.extended"
    CYCLE_1_1 = "cycle1.1"
    CYCLE_1_2 = "cycle1.2"
    CYCLE_1_3 = "cycle1.3"
    CYCLE_1_4 = "cycle1.4"
    CYCLE_1_5 = "cycle1.5"
    CYCLE_1_6 = "cycle1.6"
    CYCLE_2_1 = "cycle2.1"
    CYCLE_2_2 = "cycle2.2"
    CYCLE_2_3 = "cycle2.3"
    CYCLE_2_4 = "cycle2.4"
    CYCLE_2_5 = "cycle2.5"
    CYCLE_2_6 = "cycle2.6"
    CYCLE_2_7 = "cycle2.7"
    CYCLE_3_1 = "cycle3.1"
    CYCLE_3_2 = "cycle3.2"
    CYCLE_3_3 = "cycle3.3"
    CYCLE_3_4 = "cycle3.4"
    CYCLE_3_5 = "cycle3.5"
    CYCLE_3_6 = "cycle3.6"
    CYCLE_3_7 = "cycle3.7"
    CYCLE_3_8 = "cycle3.8"
    CYCLE_4_1 = "cycle4.1"
    CYCLE_4_2 = "cycle4.2"
    CYCLE_4_3 = "cycle4.3"
    CYCLE_4_4 = "cycle4.4"
    CYCLE_4_5 = "cycle4.5"
    CYCLE_4_6 = "cycle4.6"
    CYCLE_4_7 = "cycle4.7"
    CYCLE_4_8 = "cycle4.8"
    CYCLE_4_9 = "cycle4.9"
    CYCLE_5_1 = "cycle5.1"
    CYCLE_5_2 = "cycle5.2"
    CYCLE_5_3 = "cycle5.3"
    CYCLE_5_4 = "cycle5.4"
    CYCLE_5_5 = "cycle5.5"
    CYCLE_5_6 = "cycle5.6"
    CYCLE_5_7 = "cycle5.7"
    CYCLE_5_8 = "cycle5.8"
    CYCLE_5_9 = "cycle5.9"
    CYCLE_5_10 = "cycle6.10"
    CYCLE_6_1 = "cycle6.1"
    CYCLE_6_2 = "cycle6.2"
    CYCLE_6_3 = "cycle6.3"
    CYCLE_6_4 = "cycle6.4"
    CYCLE_6_5 = "cycle6.5"
    CYCLE_6_6 = "cycle6.6"
    CYCLE_6_7 = "cycle6.7"
    CYCLE_6_8 = "cycle6.8"
    CYCLE_6_9 = "cycle6.9"
    CYCLE_6_10 = "cycle6.10"
    CYCLE_7_1 = "cycle7.1"
    CYCLE_7_2 = "cycle7.2"
    CYCLE_7_3 = "cycle7.3"
    CYCLE_7_4 = "cycle7.4"
    CYCLE_7_5 = "cycle7.5"
    CYCLE_7_6 = "cycle7.6"
    CYCLE_7_7 = "cycle7.7"
    CYCLE_7_8 = "cycle7.8"
    CYCLE_7_9 = "cycle7.9"
    CYCLE_7_10 = "cycle7.10"
    CYCLE_8_1 = "cycle8.1"
    CYCLE_8_2 = "cycle8.2"
    CYCLE_8_3 = "cycle8.3"
    CYCLE_8_4 = "cycle8.4"
    CYCLE_8_5 = "cycle8.5"
    CYCLE_8_6 = "cycle8.6"
    CYCLE_8_7 = "cycle8.7"
    CYCLE_8_8 = "cycle8.8"
    CYCLE_8_9 = "cycle8.9"
    CYCLE_8_10 = "cycle8.10"
    OUT_01 = "out01"
    OUT_02 = "out02"
    OUT_03 = "out03"
    OUT_04 = "out04"
    OUT_05 = "out05"
    OUT_06 = "out06"
    OUT_07 = "out07"
    OUT_08 = "out08"
    OUT_09 = "out09"
    OUT_10 = "out10"
    OUT_11 = "out11"
    OUT_12 = "out12"
    OUT_13 = "out13"
    OUT_14 = "out14"
    OUT_15 = "out15"
    OUT_16 = "out16"
    OUT_17 = "out17"
    OUT_18 = "out18"
    OUT_19 = "out19"
    OUT_20 = "out20"
    OUT_21 = "out21"
    OUT_22 = "out22"
    OUT_23 = "out23"
    OUT_24 = "out24"
    OUT_25 = "out25"
    OUT_26 = "out26"
    OUT_27 = "out27"
    OUT_28 = "out28"


class ACAVersions(enum.Enum):
    ALL = "all"
    CYCLE_1 = "cycle1"
    CYCLE_2_i = "cycle2.i"
    CYCLE_2_ns = "cycle2.ns"
    CYCLE_3 = "cycle3"
    CYCLE_4 = "cycle4"
    CYCLE_5 = "cycle5"
    CYCLE_6 = "cycle6"
    CYCLE_7 = "cycle7"
    CYCLE_7_named = "cycle7.named"
    CYCLE_8 = "cycle8"
    CYCLE_8_named = "cycle8.named"
    i = "i"
    ns = "ns"
    tp = "tp"


class ATCAVersions(enum.Enum):
    ALL = "all"
    _1_5a = "1.5a"
    _1_5b = "1.5b"
    _1_5c = "1.5c"
    _1_5d = "1.5d"
    _6a = "6a"
    _6b = "6b"
    _6c = "6c"
    _6d = "6d"
    _122c = "122c"
    _750a = "750a"
    _750b = "750b"
    _750c = "750c"
    _750d = "750d"
    _ew214 = "ew214"
    _ew352 = "ew352"
    _ew367 = "ew367"
    _h75 = "h75"
    _h168 = "h168"
    _h214 = "h214"
    _ns214 = "ns214"


class CARMAVersions(enum.Enum):
    """
    Combined Array for Research in Millimeter-wave Astronomy
    telescope configuration names
    """

    A = "a"
    B = "b"
    C = "c"
    D = "d"
    E = "e"


class NGVLAVersions(enum.Enum):
    """
    next-generation Very Large Array telescope configuration names
    """

    CORE_rev_B = "core-revB"
    CORE_rev_C = "core-revC"
    gb_vlba_rev_B = "gb-vlba-revB"
    lba_rev_C = "lba-revC"
    main_rev_C = "main-revC"
    mid_subarray_rev_C = "mid-subarray-revC"
    plains_rev_B = "plains-revB"
    plains_rev_C = "plains-revC"
    rev_B = "revB"
    rev_C = "revC"
    sba_rev_B = "sba-revB"
    sba_rev_C = "sba-revC"


class PDBIVersions(enum.Enum):
    A = "a"
    B = "b"
    C = "c"
    D = "d"


class SKAMidAA0Point5Versions(enum.Enum):
    SKA_OST_ARRAY_CONFIG_2_3_1 = "ska-ost-array-config-2.3.1"


class SKAMidAA1Versions(enum.Enum):
    SKA_OST_ARRAY_CONFIG_2_3_1 = "ska-ost-array-config-2.3.1"


class SKAMidAA2Versions(enum.Enum):
    SKA_OST_ARRAY_CONFIG_2_3_1 = "ska-ost-array-config-2.3.1"


class SKAMidAAStarVersions(enum.Enum):
    SKA_OST_ARRAY_CONFIG_2_3_1 = "ska-ost-array-config-2.3.1"


class SKAMidAA4Versions(enum.Enum):
    SKA_OST_ARRAY_CONFIG_2_3_1 = "ska-ost-array-config-2.3.1"


class SKALowAA0Point5Versions(enum.Enum):
    SKA_OST_ARRAY_CONFIG_2_3_1 = "ska-ost-array-config-2.3.1"


class SKALowAA1Versions(enum.Enum):
    SKA_OST_ARRAY_CONFIG_2_3_1 = "ska-ost-array-config-2.3.1"


class SKALowAA2Versions(enum.Enum):
    SKA_OST_ARRAY_CONFIG_2_3_1 = "ska-ost-array-config-2.3.1"


class SKALowAAStarVersions(enum.Enum):
    SKA_OST_ARRAY_CONFIG_2_3_1 = "ska-ost-array-config-2.3.1"


class SKALowAA4Versions(enum.Enum):
    SKA_OST_ARRAY_CONFIG_2_3_1 = "ska-ost-array-config-2.3.1"


class SMAVersions(enum.Enum):
    COMPACT_N = "compact.n"
    COMPACT = "compact"
    EXTENDED = "extended"
    SUBCOMPACT = "subcompact"
    VEXTENDED = "vextended"


class VLAVersions(enum.Enum):
    A = "a"
    B = "b"
    C = "c"
    D = "d"
    B_n_a = "bna"
    C_n_b = "cnb"
    D_n_c = "dnc"


class MWAVersion(enum.Enum):
    ONE = "1"
    TWO_COMPACT = "2compact"
    TWO_EXTENDED = "2ext"


# ska low revision 02 2016-05-31
