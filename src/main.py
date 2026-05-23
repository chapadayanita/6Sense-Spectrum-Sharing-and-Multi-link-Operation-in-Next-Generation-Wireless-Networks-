# ==========================================
# AI-DRIVEN SPECTRUM ALLOCATION
# WITH SPECTRUM SHARING (FAIR RB SHARING)
# ==========================================

# ==========================================
# IMPORTS
# ==========================================

import sqlite3
import random
import math
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# ==========================================
# DATASET PATH
# ==========================================

DATA_PATH = "/content/drive/MyDrive"

# ==========================================
# LOAD JSON
# ==========================================

def load_json(filename):

    path = os.path.join(DATA_PATH, filename)

    with open(path, 'r') as f:

        data = json.load(f)

    return data

# ==========================================
# LOAD DATASETS
# ==========================================

sd_traffic = load_json(
    '24_TrafficSD_short64.json'
)

sd_user = load_json(
    'TrafficSD_user64.json'
)

sd_poi = load_json(
    'TrafficSD_poi64.json'
)

# ==========================================

nc_traffic = load_json(
    '24_TrafficNC_short64.json'
)

nc_user = load_json(
    'TrafficNC_user64.json'
)

nc_poi = load_json(
    'TrafficNC_poi64.json'
)

# ==========================================

nj_traffic = load_json(
    '24_TrafficNJ_short64.json'
)

nj_user = load_json(
    'TrafficNJ_user64.json'
)

nj_poi = load_json(
    'TrafficNJ_poi64.json'
)

print("\nALL DATASETS LOADED")

# ==========================================
# MODEL
# ==========================================

class ConvLSTMCell(nn.Module):

    def __init__(

        self,

        input_dim,

        hidden_dim,

        kernel_size=3
    ):

        super().__init__()

        padding = kernel_size // 2

        self.hidden_dim = hidden_dim

        self.conv = nn.Conv2d(

            input_dim + hidden_dim,

            4 * hidden_dim,

            kernel_size,

            padding=padding
        )

    def forward(

        self,

        x,

        h_prev,

        c_prev
    ):

        combined = torch.cat(

            [x, h_prev],

            dim=1
        )

        conv_output = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.chunk(

            conv_output,

            4,

            dim=1
        )

        i = torch.sigmoid(cc_i)

        f = torch.sigmoid(cc_f)

        o = torch.sigmoid(cc_o)

        g = torch.tanh(cc_g)

        c = f * c_prev + i * g

        h = o * torch.tanh(c)

        return h, c


class UniversalTrafficModel(nn.Module):

    def __init__(self):

        super().__init__()

        # CNN

        self.cnn = nn.Sequential(

            nn.Conv2d(6, 16, 3, padding=1),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.Conv2d(16, 32, 3, padding=1),

            nn.BatchNorm2d(32),

            nn.ReLU()
        )

        # TEMPORAL

        self.temporal = nn.Sequential(

            nn.Conv1d(

                32,

                32,

                kernel_size=3,

                padding=1
            ),

            nn.ReLU(),

            nn.Conv1d(

                32,

                32,

                kernel_size=3,

                padding=1
            ),

            nn.ReLU()
        )

        # CONVLSTM

        self.convlstm = ConvLSTMCell(

            input_dim=32,

            hidden_dim=32
        )

        # ATTENTION

        self.attention = nn.MultiheadAttention(

            embed_dim=32,

            num_heads=4,

            batch_first=True
        )

        # SHARED

        self.shared = nn.Sequential(

            nn.Linear(32, 64),

            nn.ReLU()
        )

        # HEADS

        self.short_head = nn.Linear(

            64,

            16 * 4 * 4
        )

        self.long_head = nn.Linear(

            64,

            16 * 4 * 4
        )

        self.generation_head = nn.Linear(

            64,

            16 * 4 * 4
        )

    def forward(self, x):

        B, T, H, W, C = x.shape

        features = []

        for t in range(T):

            xt = x[:, t]

            xt = xt.permute(

                0,

                3,

                1,

                2
            )

            feat = self.cnn(xt)

            pooled = torch.mean(

                feat,

                dim=[2,3]
            )

            features.append(pooled)

        features = torch.stack(

            features,

            dim=1
        )

        temp = features.permute(

            0,

            2,

            1
        )

        temp = self.temporal(temp)

        temp = temp.permute(

            0,

            2,

            1
        )

        attn_out, _ = self.attention(

            temp,

            temp,

            temp
        )

        final = attn_out[:, -1]

        shared = self.shared(final)

        short = self.short_head(shared)

        long = self.long_head(shared)

        gen = self.generation_head(shared)

        short = short.reshape(

            B,

            16,

            4,

            4
        )

        long = long.reshape(

            B,

            16,

            4,

            4
        )

        gen = gen.reshape(

            B,

            16,

            4,

            4
        )

        return short, long, gen

# ==========================================
# LOAD MODEL
# ==========================================

device = torch.device(

    'cuda'

    if torch.cuda.is_available()

    else 'cpu'
)

model = UniversalTrafficModel().to(device)

model.load_state_dict(

    torch.load(

        'best_universal_model.pt',

        map_location=device
    ),

    strict=False
)

model.eval()

print("\nAI TRAFFIC MODEL LOADED")

# ==========================================
# DATABASE
# ==========================================

conn = sqlite3.connect(
    "spectrum_pf.db"
)

cur = conn.cursor()

cur.execute(
    "DROP TABLE IF EXISTS logs"
)

cur.execute("""

CREATE TABLE logs (

    time_slot INTEGER,

    mno TEXT,

    num_requested INTEGER,

    channels_requested TEXT,

    allocated_channels TEXT,

    allocated_rbs TEXT,

    shared_rbs TEXT,

    status TEXT,

    expiry_time INTEGER
)

""")

conn.commit()

# ==========================================
# USER INPUTS
# ==========================================

num_mnos = int(
    input("Enter number of MNOs: ")
)

num_channels = int(
    input("Enter number of channels: ")
)

rbs_per_channel = int(
    input("Enter RBs per channel: ")
)

time_slots = int(
    input("Enter number of time slots: ")
)

print("\nENTER MNO DETAILS")

mno_priority = {}

mno_users_count = {}

for i in range(num_mnos):

    name = input(
        f"MNO {i+1} name: "
    )

    pr = int(
        input(
            f"Priority for {name} (lower = higher): "
        )
    )

    mno_priority[name] = pr

    ucount = int(
        input(
            f"Number of users for {name}: "
        )
    )

    mno_users_count[name] = ucount

# ==========================================
# CHANNELS
# ==========================================

channels = {

    ch: {

        "owner": None,

        "expiry": 0
    }

    for ch in range(

        1,

        num_channels + 1
    )
}

LEASE = 3

# ==========================================
# USERS
# ==========================================

users = []

avg_thr = {}

BS_X, BS_Y = 0, 0

PATH_LOSS_EXP = 2.5

user_id = 1

for mno, count in mno_users_count.items():

    for _ in range(count):

        x = random.randint(0, 500)

        y = random.randint(0, 500)

        dist = math.sqrt(

            (x - BS_X)**2 +

            (y - BS_Y)**2

        ) + 1

        users.append({

            "id": user_id,

            "mno": mno,

            "dist": dist
        })

        avg_thr[user_id] = 1

        user_id += 1

# ==========================================
# EXTRACT ARRAYS
# ==========================================

sd_traffic_arr = np.array(
    sd_traffic['X_train']
)[0]

sd_user_arr = np.array(
    sd_user['X_train']
)[0]

sd_poi_arr = np.array(
    sd_poi['X_train']
)

# ==========================================

nc_traffic_arr = np.array(
    nc_traffic['X_train']
)[0]

nc_user_arr = np.array(
    nc_user['X_train']
)[0]

nc_poi_arr = np.array(
    nc_poi['X_train']
)

# ==========================================

nj_traffic_arr = np.array(
    nj_traffic['X_train']
)[0]

nj_user_arr = np.array(
    nj_user['X_train']
)[0]

nj_poi_arr = np.array(
    nj_poi['X_train']
)

# ==========================================
# TAKE FIRST 48 TIMESTEPS
# ==========================================

sd_traffic_arr = sd_traffic_arr[:, :48]
sd_user_arr    = sd_user_arr[:, :48]

nc_traffic_arr = nc_traffic_arr[:, :48]
nc_user_arr    = nc_user_arr[:, :48]

nj_traffic_arr = nj_traffic_arr[:, :48]
nj_user_arr    = nj_user_arr[:, :48]

# ==========================================
# ADD CHANNEL DIMENSION
# ==========================================

sd_traffic_arr = sd_traffic_arr[..., np.newaxis]
sd_user_arr    = sd_user_arr[..., np.newaxis]

nc_traffic_arr = nc_traffic_arr[..., np.newaxis]
nc_user_arr    = nc_user_arr[..., np.newaxis]

nj_traffic_arr = nj_traffic_arr[..., np.newaxis]
nj_user_arr    = nj_user_arr[..., np.newaxis]

# ==========================================
# REDUCE POI CHANNELS
# ==========================================

sd_poi_arr = sd_poi_arr[:, :4]
nc_poi_arr = nc_poi_arr[:, :4]
nj_poi_arr = nj_poi_arr[:, :4]

# ==========================================
# TRANSPOSE POI
# ==========================================

sd_poi_arr = np.transpose(
    sd_poi_arr,
    (0,2,3,1)
)

nc_poi_arr = np.transpose(
    nc_poi_arr,
    (0,2,3,1)
)

nj_poi_arr = np.transpose(
    nj_poi_arr,
    (0,2,3,1)
)

# ==========================================
# REPEAT POI ACROSS TIME
# ==========================================

sd_poi_arr = np.repeat(
    sd_poi_arr[:, np.newaxis],
    48,
    axis=1
)

nc_poi_arr = np.repeat(
    nc_poi_arr[:, np.newaxis],
    48,
    axis=1
)

nj_poi_arr = np.repeat(
    nj_poi_arr[:, np.newaxis],
    48,
    axis=1
)

# ==========================================
# FINAL INPUTS
# ==========================================

sd_input = np.concatenate([

    sd_traffic_arr,

    sd_user_arr,

    sd_poi_arr

], axis=-1)

nc_input = np.concatenate([

    nc_traffic_arr,

    nc_user_arr,

    nc_poi_arr

], axis=-1)

nj_input = np.concatenate([

    nj_traffic_arr,

    nj_user_arr,

    nj_poi_arr

], axis=-1)

print("\nFINAL INPUT SHAPES")

print(sd_input.shape)

print(nc_input.shape)

print(nj_input.shape)

# ==========================================
# TRAFFIC MEMORY
# ==========================================

mno_traffic_memory = {

    "a": sd_input[0],

    "b": nc_input[0],

    "c": nj_input[0]
}

city_inputs = {

    "a": sd_input,

    "b": nc_input,

    "c": nj_input
}

# ==========================================
# SINR
# ==========================================

def sinr(distance):

    P = 1000

    noise = 1e-9

    interference = 1

    signal = P / (

        distance ** PATH_LOSS_EXP
    )

    return signal / (

        interference + noise
    )

# ==========================================
# CQI + RATE
# ==========================================

SUBCARRIERS_PER_RB    = 12

OFDM_SYMBOLS_PER_SLOT = 7

SLOTS_PER_TTI         = 2

SYMBOLS_PER_RB = (

    SUBCARRIERS_PER_RB *

    OFDM_SYMBOLS_PER_SLOT *

    SLOTS_PER_TTI
)

def cqi_map(sinr_db):

    if sinr_db < 0:

        return 1, 2

    elif sinr_db < 5:

        return 3, 2

    elif sinr_db < 10:

        return 7, 4

    elif sinr_db < 20:

        return 10, 6

    else:

        return 15, 8

def cqi_to_mcs_bits(cqi):

    if cqi <= 3:

        return 2

    elif cqi <= 7:

        return 4

    elif cqi <= 10:

        return 6

    else:

        return 8

def bits_per_rb(cqi):

    bits_per_symbol = cqi_to_mcs_bits(cqi)

    return SYMBOLS_PER_RB * bits_per_symbol

def rate(cqi, rbs):

    coding_rate = 0.85

    return (

        rbs *

        bits_per_rb(cqi) *

        coding_rate
    )

# ==========================================
# EXPIRE
# ==========================================

def expire(t):

    for ch in channels:

        if channels[ch]["expiry"] <= t:

            channels[ch]["owner"] = None

# ==========================================
# CHANNEL ALLOCATION
# ==========================================

def ch_allocate(req, t):

    for m in sorted(

        mno_priority,

        key=lambda x: mno_priority[x]
    ):

        for ch in req[m]:

            owner = channels[ch]["owner"]

            if owner is None:

                channels[ch]["owner"] = m

                channels[ch]["expiry"] = t + LEASE

                continue

            if owner == m:

                continue

            if mno_priority[m] < mno_priority[owner]:

                channels[ch]["owner"] = m

                channels[ch]["expiry"] = t + LEASE

# ==========================================
# AI-DRIVEN REQUEST GENERATION
# ==========================================

def generate_requests(

    active_channels_by_mno,

    future_traffic
):

    req = {}

    traffic_scores = future_traffic.mean(axis=0)

    flat_scores = traffic_scores.flatten()

    ranked_channels = np.argsort(
        flat_scores
    )

    ranked_channels = ranked_channels[::-1]

    for m in active_channels_by_mno:

        available = [

            ch

            for ch in range(

                1,

                num_channels + 1
            )

            if ch not in active_channels_by_mno[m]
        ]

        if len(available) == 0:

            req[m] = []

            continue

        demand = max(

            1,

            int(

                np.mean(future_traffic)

                * num_channels
            )
        )

        selected = []

        for idx in ranked_channels:

            ch = (

                idx % num_channels

            ) + 1

            if ch in available:

                if ch not in selected:

                    selected.append(int(ch))

            if len(selected) >= demand:

                break

        if len(selected) == 0:

            selected.append(
                random.choice(available)
            )

        req[m] = selected

    return req

# ==========================================
# AVERAGE CQI FOR MNO
# ==========================================

def compute_avg_cqi(mno):

    mno_users = [
        u for u in users if u["mno"] == mno
    ]

    if not mno_users:

        return 7

    cqi_sum = 0

    for u in mno_users:

        s = sinr(u["dist"])

        sinr_db = 10 * math.log10(s)

        cqi, _ = cqi_map(sinr_db)

        cqi_sum += cqi

    return cqi_sum / len(mno_users)

# ==========================================
# DEMAND-BASED RB COUNT FROM TRAFFIC
# ==========================================

def compute_rb_demand(mno, future_traffic, total_rbs):

    traffic_mean = float(np.mean(future_traffic))

    avg_cqi = compute_avg_cqi(mno)

    traffic_normalized = min(traffic_mean / 10.0, 1.0)

    cqi_normalized = avg_cqi / 15.0

    demand_ratio = traffic_normalized / (cqi_normalized + 1e-6)

    demanded = int(math.ceil(demand_ratio * total_rbs))

    demanded = max(1, min(demanded, total_rbs))

    return demanded

# ==========================================
# PF SCHEDULER
# ==========================================

def pf_schedule(mno, channel_rbs):

    eligible = [

        u

        for u in users

        if u["mno"] == mno
    ]

    if not eligible:

        return []

    allocation = []

    for rb in channel_rbs:

        best_user = None

        best_pf   = -1

        best_cqi  = None

        best_rate = None

        for u in eligible:

            s = sinr(u["dist"])

            sinr_db = 10 * math.log10(s)

            cqi, _ = cqi_map(sinr_db)

            r = rate(cqi, 1)

            pf = r / avg_thr[u["id"]]

            if pf > best_pf:

                best_pf   = pf

                best_user = u

                best_cqi  = cqi

                best_rate = r

        allocation.append(

            (

                best_user,

                rb,

                best_cqi,

                best_rate
            )
        )

        avg_thr[best_user["id"]] = (

            0.8 * avg_thr[best_user["id"]]

            + 0.2 * best_rate
        )

    return allocation

# ==========================================
# TRACK RB USAGE PER SLOT FOR PLOTTING
# ==========================================

mno_own_rbs_per_slot    = {m: [] for m in mno_priority}

mno_shared_rbs_per_slot = {m: [] for m in mno_priority}

# ==========================================
# MAIN LOOP
# ==========================================

user_throughput = {

    u["id"]: 0

    for u in users
}

for t in range(1, time_slots + 1):

    print(f"\nTIME SLOT {t}")

    # ======================================
    # AI TRAFFIC PREDICTION
    # ======================================

    future_traffic_by_mno = {}

    for mno in mno_priority:

        model_input = torch.FloatTensor(

            mno_traffic_memory[mno]

        ).unsqueeze(0).to(device)

        with torch.no_grad():

            pred_short, pred_long, pred_gen = model(
                model_input
            )

        future_traffic = pred_short.cpu().numpy()[0]

        future_traffic_by_mno[mno] = future_traffic

        print(

            f"\n{mno} Predicted Traffic Mean:",

            np.mean(future_traffic)
        )

    # ======================================
    # EXPIRE CHANNELS
    # ======================================

    expire(t)

    active = {

        m: set()

        for m in mno_priority
    }

    for ch in channels:

        if channels[ch]["owner"] is not None:

            active[
                channels[ch]["owner"]
            ].add(ch)

    # ======================================
    # AI REQUESTS
    # ======================================

    req = {}

    for mno in mno_priority:

        partial_req = generate_requests(

            {
                mno: active[mno]
            },

            future_traffic_by_mno[mno]
        )

        req[mno] = partial_req[mno]

    # ======================================
    # CHANNEL ALLOCATION
    # ======================================

    ch_allocate(req, t)

    alloc_info = {

        m: {

            "ch":        [],

            "rb":        [],

            "shared_rb": []
        }

        for m in mno_priority
    }

    # ======================================
    # PRIMARY RB ALLOCATION (DEMAND-BASED)
    # ======================================
    # Each MNO uses RBs from its OWN channel
    # based on AI predicted traffic demand.
    # Leftover RBs go into shared pool.
    # ======================================

    shared_pool = []

    total_system_rbs = num_channels * rbs_per_channel

    for ch in channels:

        owner = channels[ch]["owner"]

        if owner is None:

            continue

        alloc_info[owner]["ch"].append(ch)

        start = (ch - 1) * rbs_per_channel + 1

        all_rbs = list(
            range(start, start + rbs_per_channel)
        )

        demanded_rbs = compute_rb_demand(

            owner,

            future_traffic_by_mno[owner],

            rbs_per_channel
        )

        used_rbs     = all_rbs[:demanded_rbs]

        leftover_rbs = all_rbs[demanded_rbs:]

        alloc = pf_schedule(owner, used_rbs)

        for u, rb, cqi, r in alloc:

            user_throughput[u["id"]] += r

            alloc_info[owner]["rb"].append(rb)

        for rb in leftover_rbs:

            shared_pool.append((rb, owner))

        print(

            f"\nChannel {ch} owner={owner} | "

            f"AI demanded={demanded_rbs} RBs | "

            f"used={len(used_rbs)} | "

            f"leftover={len(leftover_rbs)} → shared pool"
        )

    # ======================================
    # FAIR SPECTRUM SHARING
    # ======================================
    # Logic:
    # 1. MNO first uses OWN RBs
    # 2. Compute unmet demand for each MNO
    # 3. Split shared pool PROPORTIONALLY
    #    by priority weight — higher priority
    #    gets larger share, but every MNO
    #    with unmet demand gets something
    # 4. Each MNO borrows ONLY its allocated
    #    share (not greedy all-or-nothing)
    # ======================================

    if shared_pool:

        print(
            f"\nSHARED POOL: {len(shared_pool)} RBs available"
        )

        # ----------------------------------
        # STEP 1: Compute unmet demand
        # ----------------------------------

        mno_unmet_demand = {}

        for mno in mno_priority:

            own_rbs = len(
                alloc_info[mno]["rb"]
            )

            total_demanded = compute_rb_demand(
                mno,
                future_traffic_by_mno[mno],
                total_system_rbs
            )

            unmet = max(0, total_demanded - own_rbs)

            mno_unmet_demand[mno] = unmet

        print("\nUnmet RB demands after primary allocation:")

        for mno in mno_unmet_demand:

            total_demanded = compute_rb_demand(
                mno,
                future_traffic_by_mno[mno],
                total_system_rbs
            )

            print(
                f"  {mno} | "
                f"own_rbs={len(alloc_info[mno]['rb'])} | "
                f"total_demand={total_demanded} | "
                f"unmet={mno_unmet_demand[mno]}"
            )

        # ----------------------------------
        # STEP 2: Priority weights
        # Lower priority NUMBER = higher weight
        # e.g. priority 1 → weight 3
        #      priority 2 → weight 2
        #      priority 3 → weight 1
        # ----------------------------------

        max_priority = max(mno_priority.values())

        mno_weights = {
            mno: (max_priority - mno_priority[mno] + 1)
            for mno in mno_priority
            if mno_unmet_demand[mno] > 0
        }

        total_weight = sum(mno_weights.values())

        pool_size = len(shared_pool)

        # ----------------------------------
        # STEP 3: Allocate share per MNO
        # proportional to weight,
        # capped by actual unmet demand
        # ----------------------------------

        mno_pool_quota = {}

        for mno in mno_weights:

            if total_weight == 0:

                quota = 0

            else:

                quota = int(
                    math.floor(
                        (mno_weights[mno] / total_weight)
                        * pool_size
                    )
                )

            # Cap at actual unmet demand
            quota = min(quota, mno_unmet_demand[mno])

            mno_pool_quota[mno] = quota

        # ----------------------------------
        # Distribute any leftover RBs
        # (from floor rounding) to highest
        # priority MNOs first
        # ----------------------------------

        assigned_total = sum(mno_pool_quota.values())

        leftover_quota = pool_size - assigned_total

        priority_order = sorted(
            mno_priority,
            key=lambda x: mno_priority[x]
        )

        for mno in priority_order:

            if leftover_quota <= 0:

                break

            if mno not in mno_pool_quota:

                continue

            still_needs = (
                mno_unmet_demand[mno]
                - mno_pool_quota[mno]
            )

            if still_needs > 0:

                extra = min(still_needs, leftover_quota)

                mno_pool_quota[mno] += extra

                leftover_quota -= extra

        print("\nPriority-weighted shared pool quotas:")

        for mno in priority_order:

            quota = mno_pool_quota.get(mno, 0)

            weight = mno_weights.get(mno, 0)

            print(
                f"  {mno} | "
                f"priority={mno_priority[mno]} | "
                f"weight={weight} | "
                f"quota={quota} RBs"
            )

        # ----------------------------------
        # STEP 4: Each MNO borrows ONLY
        # its quota from shared pool
        # ----------------------------------

        shared_pool_remaining = list(shared_pool)

        for mno in priority_order:

            quota = mno_pool_quota.get(mno, 0)

            if quota == 0:

                if mno_unmet_demand.get(mno, 0) == 0:

                    print(
                        f"\n{mno}: "
                        f"own RBs sufficient, "
                        f"skipping borrowing"
                    )

                else:

                    print(
                        f"\n{mno}: "
                        f"no pool quota assigned "
                        f"(no eligible RBs)"
                    )

                continue

            if not shared_pool_remaining:

                print(
                    f"\n{mno}: shared pool exhausted"
                )

                break

            borrowed        = []
            still_remaining = []

            for (rb, rb_owner) in shared_pool_remaining:

                # Cannot borrow own leftovers
                if rb_owner == mno:

                    still_remaining.append(
                        (rb, rb_owner)
                    )

                    continue

                # Borrow only up to quota
                if len(borrowed) < quota:

                    borrowed.append(rb)

                else:

                    still_remaining.append(
                        (rb, rb_owner)
                    )

            shared_pool_remaining = still_remaining

            # PF Scheduling on borrowed RBs

            if borrowed:

                alloc = pf_schedule(
                    mno,
                    borrowed
                )

                for u, rb, cqi, r in alloc:

                    user_throughput[u["id"]] += r

                    alloc_info[mno]["shared_rb"].append(rb)

                print(
                    f"\n{mno} (priority={mno_priority[mno]}) "
                    f"quota={quota} | "
                    f"borrowed {len(borrowed)} RBs "
                    f"from shared pool: {borrowed}"
                )

            else:

                print(
                    f"\n{mno}: "
                    f"no eligible shared RBs available"
                )

    else:

        print("\nNo shared pool RBs available this slot")

    # ======================================
    # DATABASE LOGGING
    # ======================================

    for m in req:

        requested  = req[m]

        allocated  = alloc_info[m]["ch"]

        shared_rbs = alloc_info[m]["shared_rb"]

        status = (
          "GRANTED"
          if (
            len(allocated) > 0
            or len(shared_rbs) > 0
        )
        else "PENDING"
)

        expiry_time = (

            t + LEASE

            if status == "GRANTED"

            else -1
        )

        cur.execute("""

            INSERT INTO logs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)

        """, (

            t,

            m,

            len(requested),

            str(requested),

            str(allocated),

            str(alloc_info[m]["rb"]),

            str(shared_rbs),

            status,

            expiry_time
        ))

    conn.commit()

    # ======================================
    # PRINT CHANNEL STATUS
    # ======================================

    print("\nCHANNEL ALLOCATION STATUS")

    for ch in channels:

        print(

            f"Channel {ch} -> "

            f"{channels[ch]['owner']}"
        )

    # ======================================
    # PRINT SHARING SUMMARY
    # ======================================

    print("\nSPECTRUM SHARING SUMMARY")

    for mno in mno_priority:

        own_rbs    = alloc_info[mno]["rb"]

        shared_rbs = alloc_info[mno]["shared_rb"]

        total_rbs  = len(own_rbs) + len(shared_rbs)

        print(

            f"{mno} (priority={mno_priority[mno]}): "

            f"own_rbs={len(own_rbs)} | "

            f"shared_rbs_borrowed={len(shared_rbs)} | "

            f"total_rbs={total_rbs}"
        )

    # ======================================
    # TRACK RB COUNTS FOR PLOT
    # ======================================

    for mno in mno_priority:

        mno_own_rbs_per_slot[mno].append(
            len(alloc_info[mno]["rb"])
        )

        mno_shared_rbs_per_slot[mno].append(
            len(alloc_info[mno]["shared_rb"])
        )

    # ======================================
    # REAL-TIME UPDATE
    # ======================================

    for mno in mno_priority:

        city_data = city_inputs[mno]

        next_index = (t) % len(city_data)

        next_state = city_data[next_index]

        mno_traffic_memory[mno] = next_state

# ==========================================
# PLOT: RB USAGE PER MNO PER TIME SLOT
# Solid bar  = Own channel RBs (primary)
# Hatched // = Shared pool RBs (borrowed)
# ==========================================

time_axis = list(range(1, time_slots + 1))

x         = np.arange(len(time_axis))

bar_width = 0.8 / len(mno_priority)

fig, ax = plt.subplots(figsize=(12, 5))

colors = plt.cm.tab10.colors

for i, mno in enumerate(sorted(

    mno_priority,

    key=lambda x: mno_priority[x]
)):

    own_vals    = mno_own_rbs_per_slot[mno]

    shared_vals = mno_shared_rbs_per_slot[mno]

    offset = (

        (i - len(mno_priority) / 2) * bar_width

        + bar_width / 2
    )

    color = colors[i % len(colors)]

    # Own RBs - solid bar
    ax.bar(

        x + offset,

        own_vals,

        width=bar_width,

        label=f"{mno} (priority={mno_priority[mno]}) Own RBs",

        color=color,

        alpha=0.9
    )

    # Shared RBs - stacked on top with hatch
    ax.bar(

        x + offset,

        shared_vals,

        width=bar_width,

        bottom=own_vals,

        label=f"{mno} Shared RBs (borrowed)",

        color=color,

        alpha=0.5,

        hatch="//"
    )

ax.set_title(

    "RB Usage Per MNO Per Time Slot\n"

    "(Solid = Own Channel RBs  |  Hatched // = Shared Pool RBs Borrowed)",

    fontsize=12
)

ax.set_xlabel("Time Slot")

ax.set_ylabel("Number of RBs")

ax.set_xticks(x)

ax.set_xticklabels([f"T{t}" for t in time_axis])

ax.legend(loc="upper right", fontsize=8)

ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()

plt.show()

# ==========================================
# THROUGHPUT PLOTS
# ==========================================

for mno in mno_priority:

    mno_users = [

        u

        for u in users

        if u["mno"] == mno
    ]

    mno_users = sorted(

        mno_users,

        key=lambda x: x["id"]
    )

    user_ids = [

        u["id"]

        for u in mno_users
    ]

    throughput = [

        user_throughput[u["id"]]

        for u in mno_users
    ]

    plt.figure()

    plt.bar(
        user_ids,
        throughput
    )

    plt.title(
        f"User Throughput - {mno}"
    )

    plt.xlabel("User ID")

    plt.ylabel("Throughput")

    plt.grid(axis='y')

    plt.show()

# ==========================================
# EXPORT CSV
# ==========================================

df = pd.read_sql_query(
    "SELECT * FROM logs",
    conn
)

df.to_csv(
    "spectrum_logs.csv",
    index=False
)

print("\nspectrum_logs.csv GENERATED")

conn.close()