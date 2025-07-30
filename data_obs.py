# data_obs.py
import numpy as np

# --- load the same dbf7.txt your retrieval uses -----------------------
d = np.genfromtxt(
    "/net/flood/home/asedai/MARGE/dbf7.txt", skip_header=17, dtype=None,
    names=["Type","MinWave","MaxWave","Depth","LoDepth","UpDepth"],
    encoding=None, invalid_raise=False
)
wl_mid = 0.5*(d["MinWave"] + d["MaxWave"])
err_mid = 0.5*(d["LoDepth"] + d["UpDepth"])
is_trans = np.char.strip(d["Type"]) == "Transmission"

# Transmission obs
w_trans = wl_mid[is_trans]
f_trans = d["Depth"][is_trans]
e_trans = err_mid[is_trans]

# Emission obs
w_emit  = wl_mid[~is_trans]
f_emit  = d["Depth"][~is_trans]
e_emit  = err_mid[~is_trans]

# Now any module can import w_trans, w_emit, etc.
