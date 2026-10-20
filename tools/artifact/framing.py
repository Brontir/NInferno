"""V3 entry/continuation headers and file payload alignment shared by reader and writer."""

import struct

MAGIC = b"NInferno"
PART_MAGIC = b"NInfPart"
HEADER = struct.Struct("<8sQ16s")
PAYLOAD_ALIGNMENT = 4096
