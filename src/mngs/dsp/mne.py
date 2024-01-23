#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-23 20:15:30 (ywatanabe)"

import mne


def to_dig_montage(channel_names):
    # Load the standard 10-20 montage
    standard_montage = mne.channels.make_standard_montage("standard_1020")

    # Get the positions of the electrodes in the standard montage
    positions = standard_montage.get_positions()

    # Filter the positions to only include the desired channels
    custom_ch_pos = {
        ch: positions["ch_pos"][ch]
        for ch in channel_names
        if ch in positions["ch_pos"]
    }

    # Create a custom DigMontage
    custom_montage = mne.channels.make_dig_montage(
        ch_pos=custom_ch_pos,
        nasion=positions["nasion"],
        lpa=positions["lpa"],
        rpa=positions["rpa"],
    )

    return custom_montage
