import enum
import functools

import torch
import torch.xpu
from torch.testing._internal.common_utils import LazyVal, TEST_XPU


XPU_ALREADY_INITIALIZED_ON_IMPORT = torch.xpu.is_initialized()


class XPUCodename(enum.Enum):
    BMG = "BMG"  # Intel® Arc™ Pro Battlemage Graphics
    LNL = "LNL"  # Intel® Core™ Ultra 200V Series Graphics
    PTL = "PTL"  # Intel® Core™ Ultra Series 3 Graphics


class XPUArch(enum.IntEnum):
    Unknown = 0
    Xe2 = 2
    Xe3 = 3


# device_id -> GPU codename
# From https://github.com/intel/intel-graphics-compiler/blob/master/inc/common/igfxfmid.h
_DEVICE_ID_TO_CODENAME = {
    0xE202: XPUCodename.BMG,
    0xE20B: XPUCodename.BMG,
    0xE20C: XPUCodename.BMG,
    0xE20D: XPUCodename.BMG,
    0xE210: XPUCodename.BMG,
    0xE211: XPUCodename.BMG,
    0xE212: XPUCodename.BMG,
    0xE215: XPUCodename.BMG,
    0xE216: XPUCodename.BMG,
    0xE220: XPUCodename.BMG,
    0xE221: XPUCodename.BMG,
    0xE222: XPUCodename.BMG,
    0xE223: XPUCodename.BMG,
    0x6420: XPUCodename.LNL,
    0x64A0: XPUCodename.LNL,
    0x64B0: XPUCodename.LNL,
    0xB080: XPUCodename.PTL,
    0xB081: XPUCodename.PTL,
    0xB082: XPUCodename.PTL,
    0xB083: XPUCodename.PTL,
    0xB084: XPUCodename.PTL,
    0xB085: XPUCodename.PTL,
    0xB086: XPUCodename.PTL,
    0xB087: XPUCodename.PTL,
    0xB08F: XPUCodename.PTL,
    0xB090: XPUCodename.PTL,
    0xB0A0: XPUCodename.PTL,
    0xB0B0: XPUCodename.PTL,
    0xB0FF: XPUCodename.PTL,
}

# GPU codename -> architecture
_CODENAME_TO_ARCH = {
    XPUCodename.BMG: XPUArch.Xe2,
    XPUCodename.LNL: XPUArch.Xe2,
    XPUCodename.PTL: XPUArch.Xe3,
}


@functools.lru_cache(1)
def get_xpu_codename() -> XPUCodename | None:
    device_id = torch.xpu.get_device_capability()["device_id"]
    return _DEVICE_ID_TO_CODENAME.get(device_id)


@functools.lru_cache(1)
def get_xpu_arch() -> XPUArch | None:
    codename = get_xpu_codename()
    return _CODENAME_TO_ARCH.get(codename, XPUArch.Unknown)


Xe2_Or_Later = LazyVal(
    lambda: torch.xpu.is_available() and get_xpu_arch() >= XPUArch.Xe2
)


def evaluate_platform_supports_flash_attention():
    if TEST_XPU:
        return Xe2_Or_Later
    return False


PLATFORM_SUPPORTS_FLASH_ATTENTION_XPU: bool = LazyVal(
    lambda: evaluate_platform_supports_flash_attention()
)

# Importing this module should NOT eagerly initialize XPU
if not XPU_ALREADY_INITIALIZED_ON_IMPORT:
    if torch.xpu.is_initialized():
        raise AssertionError("XPU should not be initialized on import")
