"""
experience_mapping.py

Builds subject-level experience labels (1=experienced, 2+ threshold varies by
dataset's own convention, 0=novice) for SKY and ds003969/Braboszcz.

OpenNeuro experience comes directly from participants.tsv's 'group' column
(group == 'expert'), applied inline in run_step0.py -- no separate function
needed there.
"""

import pandas as pd


def build_sky_experience_map(sky_path):
    """
    Returns {subj_num (1-indexed, matches SubjectMetaData.csv 'S.No.'): 0 or 1}
    Threshold: 3+ years = experienced.
    Includes manual corrections found during the original analysis:
      - subject 1 (S.No.=1) had a 0-indexed subject_id=0 collision in the old
        epoch-loading loop -- same person, same experience level, so subject 0
        is mapped identically to subject 1.
      - subject 6 (S.No.=6, epoch-loop subject_id=5) was missing from
        SubjectMetaData.csv on first pass; manually confirmed 9 years -> experienced.
    """
    sky_meta = pd.read_csv(f"{sky_path}/SubjectMetaData.csv")
    exp_map = {}
    for _, row in sky_meta.iterrows():
        subj_num = int(row["S.No."])
        years = row["MeditationExperience(in years)"]
        exp_map[subj_num] = 1 if years >= 3 else 0

    # manual corrections (see notebook cells 38, 41)
    exp_map[0] = 1   # same person as subject 1, 11 yrs, experienced
    exp_map[5] = 1   # subject 6 in file, 9 yrs, experienced -- was unmapped initially

    return exp_map


def build_ds3969_experience_map(ds3969_path):
    """
    Returns {subj_num (int, e.g. 5 for sub-005): 0 or 1}
    Threshold: 3+ years = experienced. Subjects with NaN years_of_practice are
    skipped UNLESS manually confirmed experienced via their 'group'/'notes' column.
    """
    ds3969_meta = pd.read_csv(f"{ds3969_path}/participants.tsv", sep="\t")
    exp_map = {}
    for _, row in ds3969_meta.iterrows():
        subj_num = int(row["participant_id"].split("-")[1])
        years = row["years_of_practice"]
        if pd.isna(years):
            continue  # handled by manual overrides below, or left unmapped
        exp_map[subj_num] = 1 if years >= 3 else 0

    # manual corrections (see notebook cells 38, 39) -- both NaN years but
    # confirmed experienced practitioners via group/notes columns
    exp_map[24] = 1   # htr
    exp_map[59] = 1   # tm

    return exp_map