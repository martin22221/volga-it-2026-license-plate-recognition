"""Minimal baseline-JPEG reader that recovers only the DC coefficients.

Every DCT block's DC coefficient *is* the average of that 8x8 block, so
decoding DC alone yields the image at 1/8 scale without an inverse DCT.  That
is ample for judging the average colour of a region -- which is all the plate
colour heuristic needs -- and it keeps the project free of an image-decoding
dependency.

The AC coefficients still have to be Huffman-decoded to advance the bitstream,
but they are discarded rather than transformed, and decoding can stop once the
last interesting MCU row has been read.

Scope: **baseline sequential, Huffman-coded, 8-bit** JPEG with one or three
components -- which is what this dataset contains.  Progressive, arithmetic,
12-bit and CMYK files raise :class:`JpegUnsupported` rather than being guessed
at.  Nothing here writes to disk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final, Sequence

logger = logging.getLogger(__name__)

#: Start-of-frame marker for baseline sequential DCT, the only one supported.
SOF_BASELINE: Final[int] = 0xC0

#: Start-of-frame markers we recognise but cannot decode.
SOF_UNSUPPORTED: Final[dict[int, str]] = {
    0xC1: "extended sequential",
    0xC2: "progressive",
    0xC3: "lossless",
    0xC5: "differential sequential",
    0xC6: "differential progressive",
    0xC7: "differential lossless",
    0xC9: "arithmetic extended sequential",
    0xCA: "arithmetic progressive",
    0xCB: "arithmetic lossless",
    0xCD: "arithmetic differential sequential",
    0xCE: "arithmetic differential progressive",
    0xCF: "arithmetic differential lossless",
}

_NEUTRAL_CHROMA: Final[int] = 128


class JpegUnsupported(Exception):
    """The file is a JPEG this reader deliberately does not handle."""


class JpegCorrupt(Exception):
    """The bitstream ended early or violated the format."""


@dataclass
class Component:
    """One colour component and its DC plane at block resolution."""

    identifier: int
    h_sampling: int
    v_sampling: int
    quant_table: int
    blocks_wide: int = 0
    blocks_high: int = 0
    values: list[int] | None = None

    def sample(self, x0: float, y0: float, x1: float, y1: float) -> float | None:
        """Mean value over the normalised rectangle ``(x0, y0) - (x1, y1)``.

        Coordinates are fractions of the whole image.  Returns ``None`` when
        the rectangle covers no decoded block.
        """
        if not self.values or not self.blocks_wide or not self.blocks_high:
            return None

        left = max(0, min(self.blocks_wide - 1, int(x0 * self.blocks_wide)))
        right = max(left + 1, min(self.blocks_wide, int(x1 * self.blocks_wide + 0.999)))
        top = max(0, min(self.blocks_high - 1, int(y0 * self.blocks_high)))
        bottom = max(top + 1, min(self.blocks_high, int(y1 * self.blocks_high + 0.999)))

        total = 0
        count = 0
        for row in range(top, bottom):
            base = row * self.blocks_wide
            for column in range(left, right):
                index = base + column
                if index < len(self.values):
                    total += self.values[index]
                    count += 1
        return total / count if count else None


@dataclass
class DcImage:
    """An image decoded to its DC coefficients: a 1/8-scale YCbCr thumbnail."""

    width: int
    height: int
    components: list[Component]

    @property
    def is_colour(self) -> bool:
        return len(self.components) >= 3

    def sample_ycbcr(
        self, x0: float, y0: float, x1: float, y1: float
    ) -> tuple[float, float, float] | None:
        """Mean Y, Cb, Cr over a normalised rectangle.

        A greyscale JPEG reports neutral chroma, which is truthful: it carries
        no colour information, so nothing in it can be judged yellow.
        """
        luma = self.components[0].sample(x0, y0, x1, y1)
        if luma is None:
            return None
        if not self.is_colour:
            return luma, float(_NEUTRAL_CHROMA), float(_NEUTRAL_CHROMA)

        blue = self.components[1].sample(x0, y0, x1, y1)
        red = self.components[2].sample(x0, y0, x1, y1)
        if blue is None or red is None:
            return None
        return luma, blue, red


#: Width of the Huffman lookup window. JPEG codes are at most 16 bits, so one
#: table indexed by the next 16 bits decodes any symbol in a single lookup.
_LOOKUP_BITS: Final[int] = 16
_LOOKUP_SIZE: Final[int] = 1 << _LOOKUP_BITS

#: Entries pack ``(symbol << 5) | code_length``; 0 marks an unused pattern.
_LENGTH_MASK: Final[int] = 31


def _extend(value: int, length: int) -> int:
    """Convert a JPEG magnitude-category value to its signed form."""
    if length == 0:
        return 0
    return value if value >= (1 << (length - 1)) else value - (1 << length) + 1


def _build_huffman(counts: Sequence[int], symbols: Sequence[int]) -> list[int]:
    """Build a 16-bit-indexed lookup table from a DHT segment.

    Decoding bit by bit through a dictionary is far too slow to sweep a whole
    dataset in pure Python.  Expanding each canonical code over every 16-bit
    pattern that begins with it turns symbol decoding into one list index.
    """
    table = [0] * _LOOKUP_SIZE
    code = 0
    index = 0
    for length in range(1, 17):
        for _ in range(counts[length - 1]):
            if index < len(symbols):
                entry = (symbols[index] << 5) | length
                shift = _LOOKUP_BITS - length
                start = code << shift
                for pattern in range(start, start + (1 << shift)):
                    table[pattern] = entry
                index += 1
            code += 1
        code <<= 1
    return table


def _strip_entropy(data: bytes, start: int) -> bytes:
    """Return the scan's entropy bytes with stuffing and restarts removed.

    Doing this once, up front, keeps the per-bit path free of the 0xFF00 and
    RSTn special cases.  Restart markers are dropped here and handled by
    byte-aligning at each restart interval instead.
    """
    end = len(data)
    chunks: list[bytes] = []
    position = start
    segment_start = start

    while position < end:
        byte = data[position]
        if byte != 0xFF:
            position += 1
            continue

        following = data[position + 1] if position + 1 < end else 0xD9
        if following == 0x00:
            chunks.append(data[segment_start : position + 1])  # keep the 0xFF
            position += 2
            segment_start = position
            continue
        if 0xD0 <= following <= 0xD7:
            chunks.append(data[segment_start:position])  # drop the RST marker
            position += 2
            segment_start = position
            continue
        break  # a real marker ends the scan

    chunks.append(data[segment_start:position])
    return b"".join(chunks)


def decode_dc(data: bytes, *, region_bottom: float = 1.0) -> DcImage:
    """Decode ``data`` to its DC coefficients.

    ``region_bottom`` is a fraction of image height: decoding stops once every
    MCU row above it has been read.  Pass the bottom of the area you care about
    to avoid decoding the rest of the file.
    """
    if not data.startswith(b"\xff\xd8"):
        raise JpegUnsupported("not a JPEG (no SOI marker)")

    quant_tables: dict[int, list[int]] = {}
    dc_tables: dict[int, dict[tuple[int, int], int]] = {}
    ac_tables: dict[int, dict[tuple[int, int], int]] = {}
    components: list[Component] = []
    width = height = 0
    restart_interval = 0

    position = 2
    end = len(data)
    while position + 3 < end:
        if data[position] != 0xFF:
            position += 1
            continue
        marker = data[position + 1]
        position += 2

        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            continue
        if marker == 0xD9:
            break
        if position + 1 >= end:
            break

        length = int.from_bytes(data[position : position + 2], "big")
        if length < 2:
            raise JpegCorrupt("segment length below 2")
        segment = data[position + 2 : position + length]
        segment_end = position + length

        if marker in SOF_UNSUPPORTED:
            raise JpegUnsupported(f"{SOF_UNSUPPORTED[marker]} JPEG is not supported")

        if marker == SOF_BASELINE:
            precision = segment[0]
            if precision != 8:
                raise JpegUnsupported(f"{precision}-bit samples are not supported")
            height = int.from_bytes(segment[1:3], "big")
            width = int.from_bytes(segment[3:5], "big")
            count = segment[5]
            if count not in (1, 3):
                raise JpegUnsupported(f"{count}-component JPEG is not supported")
            components = []
            for index in range(count):
                base = 6 + index * 3
                components.append(
                    Component(
                        identifier=segment[base],
                        h_sampling=segment[base + 1] >> 4,
                        v_sampling=segment[base + 1] & 15,
                        quant_table=segment[base + 2],
                    )
                )

        elif marker == 0xDB:  # DQT
            offset = 0
            while offset < len(segment):
                precision = segment[offset] >> 4
                identifier = segment[offset] & 15
                offset += 1
                size = 64 * (2 if precision else 1)
                raw = segment[offset : offset + size]
                if precision:
                    values = [
                        int.from_bytes(raw[i : i + 2], "big") for i in range(0, size, 2)
                    ]
                else:
                    values = list(raw)
                quant_tables[identifier] = values
                offset += size

        elif marker == 0xC4:  # DHT
            offset = 0
            while offset < len(segment):
                table_class = segment[offset] >> 4
                identifier = segment[offset] & 15
                counts = list(segment[offset + 1 : offset + 17])
                total = sum(counts)
                symbols = list(segment[offset + 17 : offset + 17 + total])
                table = _build_huffman(counts, symbols)
                if table_class == 0:
                    dc_tables[identifier] = table
                else:
                    ac_tables[identifier] = table
                offset += 17 + total

        elif marker == 0xDD:  # DRI
            restart_interval = int.from_bytes(segment[0:2], "big")

        elif marker == 0xDA:  # SOS
            if not components or not width or not height:
                raise JpegCorrupt("scan before frame header")
            scan_count = segment[0]
            selectors: list[tuple[int, int, int]] = []
            for index in range(scan_count):
                identifier = segment[1 + index * 2]
                tables = segment[2 + index * 2]
                selectors.append((identifier, tables >> 4, tables & 15))
            _decode_scan(
                data,
                segment_end,
                components,
                selectors,
                dc_tables,
                ac_tables,
                quant_tables,
                width,
                height,
                restart_interval,
                region_bottom,
            )
            return DcImage(width=width, height=height, components=components)

        position = segment_end

    raise JpegCorrupt("no scan found")


def _decode_scan(
    data: bytes,
    start: int,
    components: list[Component],
    selectors: Sequence[tuple[int, int, int]],
    dc_tables: dict[int, dict[tuple[int, int], int]],
    ac_tables: dict[int, dict[tuple[int, int], int]],
    quant_tables: dict[int, list[int]],
    width: int,
    height: int,
    restart_interval: int,
    region_bottom: float,
) -> None:
    if len(selectors) != len(components):
        raise JpegUnsupported("scan does not cover every component (progressive?)")

    max_h = max(component.h_sampling for component in components)
    max_v = max(component.v_sampling for component in components)
    if max_h == 0 or max_v == 0:
        raise JpegCorrupt("component declares zero sampling factor")

    mcus_wide = (width + 8 * max_h - 1) // (8 * max_h)
    mcus_high = (height + 8 * max_v - 1) // (8 * max_v)

    for component in components:
        component.blocks_wide = mcus_wide * component.h_sampling
        component.blocks_high = mcus_high * component.v_sampling
        component.values = [_NEUTRAL_CHROMA] * (
            component.blocks_wide * component.blocks_high
        )

    table_for: dict[int, tuple[int, int]] = {
        identifier: (dc_index, ac_index) for identifier, dc_index, ac_index in selectors
    }

    # Resolve every per-component constant before the hot loop.
    plan: list[tuple[list[int], list[int], int, int, int, list[int], int, int]] = []
    for component in components:
        dc_index, ac_index = table_for.get(component.identifier, (0, 0))
        dc_table = dc_tables.get(dc_index)
        ac_table = ac_tables.get(ac_index)
        if dc_table is None or ac_table is None:
            raise JpegCorrupt("scan references an undefined Huffman table")
        quant = quant_tables.get(component.quant_table)
        assert component.values is not None
        plan.append(
            (
                dc_table,
                ac_table,
                quant[0] if quant else 1,
                component.h_sampling,
                component.v_sampling,
                component.values,
                component.blocks_wide,
                component.blocks_high,
            )
        )

    stream = _strip_entropy(data, start)
    length = len(stream)
    position = 0
    buffer = 0
    count = 0
    predictors = [0] * len(plan)

    # Stop once every MCU row overlapping the region of interest is decoded.
    last_row = min(
        mcus_high,
        max(1, int(mcus_high * min(1.0, max(0.0, region_bottom))) + 1),
    )

    since_restart = 0
    for mcu_y in range(last_row):
        for mcu_x in range(mcus_wide):
            if restart_interval and since_restart == restart_interval:
                buffer = 0  # restarts are byte aligned; markers already removed
                count = 0
                predictors = [0] * len(plan)
                since_restart = 0

            for index, entry in enumerate(plan):
                (
                    dc_table,
                    ac_table,
                    scale,
                    h_sampling,
                    v_sampling,
                    values,
                    blocks_wide,
                    blocks_high,
                ) = entry

                for block_y in range(v_sampling):
                    row = mcu_y * v_sampling + block_y
                    for block_x in range(h_sampling):
                        # --- DC coefficient -------------------------------
                        while count < 16:
                            if position < length:
                                buffer = (buffer << 8) | stream[position]
                                position += 1
                            else:
                                buffer <<= 8
                            count += 8
                        code = dc_table[(buffer >> (count - 16)) & 0xFFFF]
                        size = code & _LENGTH_MASK
                        if size == 0:
                            raise JpegCorrupt("invalid DC Huffman code")
                        count -= size
                        buffer &= (1 << count) - 1
                        size = code >> 5

                        if size:
                            while count < size:
                                if position < length:
                                    buffer = (buffer << 8) | stream[position]
                                    position += 1
                                else:
                                    buffer <<= 8
                                count += 8
                            count -= size
                            raw = (buffer >> count) & ((1 << size) - 1)
                            buffer &= (1 << count) - 1
                            predictors[index] += _extend(raw, size)

                        value = predictors[index] * scale // 8 + 128
                        column = mcu_x * h_sampling + block_x
                        if row < blocks_high and column < blocks_wide:
                            values[row * blocks_wide + column] = (
                                0 if value < 0 else 255 if value > 255 else value
                            )

                        # --- AC coefficients, decoded then discarded -------
                        coefficient = 1
                        while coefficient < 64:
                            while count < 16:
                                if position < length:
                                    buffer = (buffer << 8) | stream[position]
                                    position += 1
                                else:
                                    buffer <<= 8
                                count += 8
                            code = ac_table[(buffer >> (count - 16)) & 0xFFFF]
                            bits = code & _LENGTH_MASK
                            if bits == 0:
                                raise JpegCorrupt("invalid AC Huffman code")
                            count -= bits
                            buffer &= (1 << count) - 1

                            symbol = code >> 5
                            size = symbol & 15
                            if size == 0:
                                if symbol == 0xF0:
                                    coefficient += 16
                                    continue
                                break  # end of block
                            coefficient += (symbol >> 4) + 1
                            while count < size:
                                if position < length:
                                    buffer = (buffer << 8) | stream[position]
                                    position += 1
                                else:
                                    buffer <<= 8
                                count += 8
                            count -= size
                            buffer &= (1 << count) - 1

            since_restart += 1

        if position >= length and count <= 0:
            break


def ycbcr_to_rgb(luma: float, blue: float, red: float) -> tuple[int, int, int]:
    """Convert JPEG YCbCr to 8-bit RGB, clamped."""
    r = luma + 1.402 * (red - 128)
    g = luma - 0.344136 * (blue - 128) - 0.714136 * (red - 128)
    b = luma + 1.772 * (blue - 128)
    return tuple(int(max(0, min(255, round(value)))) for value in (r, g, b))  # type: ignore[return-value]
