#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-04-27 11:04:12 (ywatanabe)"

#!/usr/bin/env python3

import itertools
import re
import sys
import warnings
from copy import deepcopy

import bids
import mngs
import numpy as np
import pandas as pd
from bids import BIDSLayout
from mne.io import read_raw_edf
from natsort import natsorted
try:
    from pandas.errors import SettingWithCopyWarning
except:
    from pandas.core.common import SettingWithCopyWarning

# from pandas.core.common import SettingWithCopyWarning
# from pandas.errors import SettingWithCopyWarning
from scipy import stats
from tqdm import tqdm

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)



def load_BIDS(BIDS_ROOT, channels, verbose=False):
    layout = BIDSLayout(BIDS_ROOT)
    entities = layout.get_entities()

    if verbose:
        layout_keys_to_print = [
            "InstitutionName",
            "PowerLineFrequency",
            "SamplingFrequency",
            "TaskName",
            "extension",
            "run",
            "session",
            "subject",
            "suffix",
            "task",
        ]

        for k in layout_keys_to_print:
            print(f"\nunique {k}s:\n{natsorted(entities[k].unique())}\n")
            sleep(1)

    # Probably, only one instituion is allowed for a BIDS dataset.
    subj_uq = natsorted(entities["subject"].unique())
    task_uq = natsorted(entities["task"].unique())
    datatype_uq = natsorted(entities["datatype"].unique())
    session_uq = natsorted(entities["session"].unique())
    run_uq = natsorted(entities["run"].unique())

    data_all = []
    for i_subj, subj in tqdm(enumerate(subj_uq)):

        try:
            # Session
            session_bids = layout.get(subject=subj, suffix="sessions")
            assert len(session_bids) == 1
            session_bids = session_bids[0]

            for task in task_uq:
                for datatype in datatype_uq:
                    for session in session_uq:
                        for run in run_uq:

                            ## EDF, signal
                            edf_bids = layout.get(
                                subject=subj,
                                session=session,
                                run=run,
                                datatype=datatype,
                                extension=".edf",
                            )
                            assert len(edf_bids) == 1
                            edf_bids = edf_bids[0]

                            ## Event
                            event_bids = layout.get(
                                subject=subj,
                                session=session,
                                run=run,
                                datatype=datatype,
                                suffix="events",
                                extension=".tsv",
                            )
                            assert len(event_bids) == 1
                            event_bids = event_bids[0]

                            ## eeg metadata from json
                            metadata_bids = layout.get(
                                subject=subj,
                                session=session,
                                run=run,
                                datatype=datatype,
                                suffix="eeg",
                                extension=".json",
                            )
                            assert len(metadata_bids) == 1
                            metadata_bids = metadata_bids[0]

                            ## Load data of a run
                            data_run = _load_a_run(
                                session_bids,
                                edf_bids,
                                event_bids,
                                metadata_bids,
                                channels,
                            )

                            ## Buffering
                            data_all.append(data_run)

        except Exception as e:
            print(f"\n{subj}:\n{e}\n")

    data_all_df = pd.concat(data_all).reset_index()
    del data_all_df["index"]

    return data_all_df


def _load_a_run(
    session_bids, edf_bids, event_bids, metadata_bids, channels, verbose=False
):
    event_df = event_bids.get_df()
    event_df["offset"] = event_df["onset"] + event_df["duration"]
    metadata_dict = metadata_bids.get_dict()

    df = event_df.copy()

    df = df[df["trial_type"] == "NoiseFreeSection"]

    eeg_data = _load_an_edf(
        edf_bids,
        channels=channels,
        starts=df["onset"],
        ends=df["offset"],
    )  # load_params["channels"], # fixme

    df["eeg"] = eeg_data

    df = pd.DataFrame(df[["eeg", "trial_type"]])

    ## Labels
    session_df = session_bids.get_df()
    for k, v in session_df.items():
        assert v.shape == (1,)
        df[k] = v[0]
    del df["session_id"]

    entities = edf_bids.get_entities()
    df["subject"] = entities["subject"]
    df["run"] = entities["run"]
    df["session"] = entities["session"]
    df["task"] = entities["task"]
    df["samp_rate"] = metadata_dict["SamplingFrequency"]
    df["channels"] = [channels for _ in range(len(df))]

    return df


def _load_an_edf(bf_edf, channels, starts=[0], ends=[None]):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mne_raw = read_raw_edf(bf_edf, verbose=False)

    sfreq = mne_raw.info["sfreq"]
    # channels = mne_raw.info["ch_names"]

    data = [
        mne_raw.get_data(picks=channels, start=int(start * sfreq), stop=int(end * sfreq))
        for start, end in zip(starts, ends)
    ]
    # data = [
    #     mne_raw.get_data(
    #         picks=channels, start=int(start * sfreq), stop=int(end * sfreq)
    #     )
    #     for start, end in zip(starts, ends)
    # ]

    return data


if __name__ == "__main__":
    import mngs

    EEG_CHANNELS = [
        "FP1",
        "F3",
        "C3",
        "P3",
        "O1",
        "FP2",
        "F4",
        "C4",
        "P4",
        "O2",
        "F7",
        "T7",
        "P7",
        "F8",
        "T8",
        "P8",
        "Fz",
        "Cz",
        "Pz",
        "A1",
        "A2",        
    ]

    BIDS_FPATH = "YOUR_BIDS_FPATH"
    BIDS_FPATH = "./data/DEM/BIDS_Osaka"
    data_all = load_BIDS(BIDS_FPATH, channels=EEG_CHANNELS)
    mngs.io.save(data_all, "./res/DEM/dataset.pkl")

    data_all.columns
    # Index(['eeg', 'trial_type', 'disease_type', 'cognitive_level', 'pet',
    #  'subject', 'run', 'session', 'task', 'samp_rate', 'channels'],
    # dtype='object')
