import numpy as np, pandas as pd
from pathlib import Path
import re

TABLE1_MEANS = {"expert": (1.85, 1.14, 0.69), "novice": (1.39, 1.59, 0.73)}   # Brandmeyer & Delorme Table 1 (Med, MW, Sleep)

def decode_probes(events_tsv, value_to_rating=None, min_gap=2.2):
    """One row per probe. Key presses are assigned to Q1/Q2/Q3 by timing; a missing press = rating 0 (HYPOTHESIS)."""
    if value_to_rating is None:
        value_to_rating = {2: 1, 4: 2, 8: 3}          # value = 2**rating; rating 0 -> value 1 -> no event
    e = pd.read_csv(events_tsv, sep="\t").sort_values("onset").reset_index(drop=True)
    stim, resp = e[e.trial_type == "stimulus"], e[e.trial_type == "response"]
    t_stim = stim.onset.values; rows = []
    for i, t0 in enumerate(t_stim):
        t1 = t_stim[i + 1] if i + 1 < len(t_stim) else np.inf
        r = resp[(resp.onset > t0) & (resp.onset < t1)]
        rows.append(dict(probe_id=i, t_probe=t0, n_resp=len(r), resp_t=r.onset.values - t0, resp_v=r.value.values))
    df = pd.DataFrame(rows)
    full = df[df.n_resp == 3]
    a = np.median([x[0] for x in full.resp_t]) if len(full) >= 3 else 2.6                            # typical Q1 latency
    g = np.median(np.concatenate([np.diff(x) for x in full.resp_t])) if len(full) >= 3 else min_gap  # typical Q-to-Q gap
    out = []
    for _, p in df.iterrows():
        n, t, v = p.n_resp, p.resp_t, p.resp_v
        q = [np.nan] * 3; slots = None; amb = False
        if n == 3: slots = [0, 1, 2]
        elif n == 2:                                    # which question has no press? pick the best timing fit
            hyp = {0: (a + g, g), 1: (a, 2 * g), 2: (a, g)}
            obs = (t[0], t[1] - t[0])
            err = {m: max(0, hyp[m][0] - obs[0])**2 * 4 + (obs[0] - hyp[m][0])**2
                      + max(0, hyp[m][1] - obs[1])**2 * 4 + (obs[1] - hyp[m][1])**2 for m in hyp}
            order = sorted(err, key=err.get); miss = order[0]
            amb = (err[order[1]] - err[order[0]]) < 1.0
            slots = [s for s in range(3) if s != miss]
        else: amb = True
        if slots is not None:
            for s, val in zip(slots, v): q[s] = value_to_rating.get(int(val), np.nan)
            for s in range(3):
                if s not in slots: q[s] = 0
        out.append(dict(probe_id=p.probe_id, t_probe=p.t_probe, n_resp=n, q1=q[0], q2=q[1], q3=q[2], ambiguous=amb))
    d = pd.DataFrame(out)
    d["label"] = np.where(d.q1 > d.q2, "MED", np.where(d.q2 > d.q1, "MW", "TIE"))
    d.loc[d.ambiguous | d[["q1", "q2"]].isna().any(axis=1), "label"] = "DROP"
    return d

def validate_decoding(per_subject, group_of):
    rows = []
    for s, d in per_subject.items():
        ok = d[~d.ambiguous]
        rows.append(dict(subject=s, group=group_of.get(s), n_probes=len(d), frac_ambiguous=d.ambiguous.mean(),
                         Med=ok.q1.mean(), MW=ok.q2.mean(), Sleep=ok.q3.mean(),
                         n_MED=(d.label == "MED").sum(), n_MW=(d.label == "MW").sum(), n_TIE=(d.label == "TIE").sum()))
    t = pd.DataFrame(rows).set_index("subject")
    return t, t.groupby("group")[["Med", "MW", "Sleep"]].mean().round(2), pd.DataFrame(TABLE1_MEANS, index=["Med", "MW", "Sleep"]).T

# ---- RUN (decodes events only, fast) ----
DATA_ROOT = "/Users/sarshanspal/Library/CloudStorage/GoogleDrive-sarahanspal@gmail.com/My Drive/meditation-eeg"

def resolve_dir(root, name):                      # Drive folder is literally named "openneuro " (trailing space)
    for p in Path(root).iterdir():
        if p.name.strip() == name.strip(): return p
    raise FileNotFoundError(f"{name!r} not found under {root}")

def brandmeyer_sessions(root, subjects):
    """Return EEG/events pairs matched by BIDS filename within each session."""
    root = Path(root)
    rows = []
    eeg_ext = {".bdf", ".edf", ".set", ".vhdr"}
    for subject in subjects:
        eegs = sorted(
            p for p in root.glob(f"{subject}/**/{subject}_*task-meditation_eeg.*")
            if p.suffix.lower() in eeg_ext
        )
        if not eegs:
            rows.append(dict(subject=subject, session="", eeg=None, events=None))
            continue
        for eeg in eegs:
            events_name = re.sub(r"_eeg\.[^.]+$", "_events.tsv", eeg.name)
            events = eeg.with_name(events_name)
            match = re.search(r"ses-[^_]+", eeg.name)
            rows.append(dict(subject=subject, session=match.group(0) if match else "",
                             eeg=eeg, events=events if events.exists() else None))
    return rows

if __name__ == "__main__":
    BM_ROOT = resolve_dir(DATA_ROOT, "openneuro ")
    pt = pd.read_csv(BM_ROOT / "participants.tsv", sep="\t")        # participant_id, gender, age, group (expert/novice)
    group_of = dict(zip(pt.participant_id, pt.group))
    sessions = brandmeyer_sessions(BM_ROOT, pt.participant_id)
    session_table = pd.DataFrame([
        {"subject": s["subject"], "session": s["session"],
         "eeg": str(s["eeg"]) if s["eeg"] else None,
         "events": str(s["events"]) if s["events"] else None,
         "has_eeg": s["eeg"] is not None, "has_events": s["events"] is not None}
        for s in sessions
    ])
    print(session_table.drop(columns=["eeg", "events"]).to_string(index=False))
    decoded = {}
    session_groups = {}
    for s in sessions:
        if s["events"] is None:
            continue
        key = f"{s['subject']}_{s['session']}" if s["session"] else s["subject"]
        decoded[key] = decode_probes(s["events"])
        session_groups[key] = group_of[s["subject"]]
    tbl, mine, paper = validate_decoding(decoded, session_groups)
    print(tbl.round(2).to_string()); print("\ndecoded group means:\n", mine); print("\npaper Table 1:\n", paper)
