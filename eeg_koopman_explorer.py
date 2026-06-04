"""
EEG KOOPMAN EXPLORER
====================
Load EEG recordings (mat / edf / npy / csv), compute per-channel spectrograms,
and explore the resulting Koopman manifold with group colour-coding and a
live group-separation score.

The spectrogram IS a time-varying Gabor transform (STFT with Hann window),
so the manifold you navigate here is literally the space GAIT predicts:
images in Gabor-frequency space, clustering by dynamical structure.

PIPELINE
--------
1. Add one or more groups by pointing at a folder of EEG files.
   Folder name becomes the group label and determines dot colour on the map.
2. Choose electrode channel (index or name if detectable).
3. Click PROCESS — spectrograms are computed, PCA manifold is built.
4. Drag the crosshair across the map; watch the reconstruction and
   the group-weight bar update live.
5. A group-separation score (centroid distance / pooled sigma) tells you
   whether the groups actually separate.

FORMATS SUPPORTED
-----------------
.mat  — raw matrix OR EEGLAB struct  (scipy)
.edf  — European Data Format         (pyedflib, optional)
.npy  — numpy array                  (numpy)
.csv  — comma-separated matrix       (numpy)

DEPENDENCIES
------------
Required : numpy  scipy  matplotlib  Pillow
Optional : pyedflib  (pip install pyedflib)   — for .edf files
           scikit-learn  (pip install scikit-learn) — fast PCA for N > 500
"""

import os
import glob
import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
from scipy import signal as sig
from scipy.io import loadmat
from PIL import Image, ImageTk

# ── optional imports ──────────────────────────────────────────────────────────
try:
    import pyedflib
    HAS_EDF = True
except ImportError:
    HAS_EDF = False

try:
    from sklearn.decomposition import PCA as SkPCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ── constants ─────────────────────────────────────────────────────────────────
CANVAS_W, CANVAS_H = 560, 510
RECON_SZ = 280
NAV_DOT = 6
GROUP_COLORS = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12',
                '#9b59b6', '#1abc9c', '#e67e22', '#e91e63']
BG, PANEL, TEXT, DIM = '#0d0d14', '#15151f', '#e0e0ee', '#666677'


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  EEG LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _first_2d(mat_dict):
    """Return the largest 2-D array from a scipy mat dict."""
    cands = {k: v for k, v in mat_dict.items()
             if not k.startswith('_')
             and isinstance(v, np.ndarray) and v.ndim == 2}
    if not cands:
        return None, None
    key = max(cands, key=lambda k: cands[k].size)
    arr = cands[key]
    return key, (arr if arr.shape[0] <= arr.shape[1] else arr.T)


def _detect_eeglab(mat_dict):
    """Try to extract data, srate, and channel labels from an EEGLAB .set/.mat."""
    for root_key in ('EEG', 'eeg', 'EEGLAB'):
        if root_key not in mat_dict:
            continue
        s = mat_dict[root_key]
        try:
            data   = np.array(s['data'][0, 0],  dtype=float)
            srate  = float(s['srate'][0, 0].flat[0])
            chans  = s['chanlocs'][0, 0]
            labels = [str(chans['labels'][0, i][0]) for i in range(chans.size)]
            if data.shape[0] > data.shape[1]:
                data = data.T
            return data, srate, labels
        except Exception:
            pass
    return None, None, None


def load_eeg(path):
    """
    Load EEG from file.  Returns (data, fs, channel_labels).
    data shape: (n_channels, n_samples)
    channel_labels: list of strings, or None.
    fs: sampling rate (float), or None if unknown.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == '.mat':
        mat = loadmat(path, squeeze_me=False, struct_as_record=True)

        # Try EEGLAB first
        data, fs, labels = _detect_eeglab(mat)
        if data is not None:
            return data, fs, labels

        # Try simple matrix keys
        for key in ('data', 'EEG', 'eeg', 'y', 'eegData', 'rawData', 'X'):
            if key in mat:
                arr = np.array(mat[key], dtype=float)
                if arr.ndim == 2:
                    if arr.shape[0] > arr.shape[1]:
                        arr = arr.T
                    return arr, None, None

        # Fallback: largest 2-D array
        _, arr = _first_2d(mat)
        if arr is not None:
            return arr.astype(float), None, None
        raise ValueError(f"No 2-D EEG array found in {os.path.basename(path)}")

    elif ext == '.edf':
        if not HAS_EDF:
            raise ImportError("Install pyedflib:  pip install pyedflib")
        f = pyedflib.EdfReader(path)
        fs    = f.getSampleFrequency(0)
        labels= f.getSignalLabels()
        data  = np.stack([f.readSignal(i) for i in range(f.signals_in_file)])
        f._close()
        return data.astype(float), float(fs), labels

    elif ext == '.npy':
        arr = np.load(path).astype(float)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[0] > arr.shape[1]:
            arr = arr.T
        return arr, None, None

    elif ext == '.csv':
        arr = np.loadtxt(path, delimiter=',').astype(float)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[0] > arr.shape[1]:
            arr = arr.T
        return arr, None, None

    else:
        raise ValueError(f"Unsupported format: {ext}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  SPECTROGRAM  (= time-varying Gabor transform)
# ═══════════════════════════════════════════════════════════════════════════════

def make_spectrogram(channel_data, fs, fmin=1.0, fmax=40.0,
                     win_sec=1.5, overlap=0.75,
                     out_h=64, out_w=64, log_power=True):
    """
    Compute STFT spectrogram of a 1-D EEG channel.
    Returns a uint8 numpy array of shape (out_h, out_w).
    Frequency axis: fmin..fmax (rows = high-freq top, low-freq bottom).
    """
    nperseg  = int(win_sec * fs)
    noverlap = int(nperseg * overlap)

    freqs, _, Sxx = sig.spectrogram(
        channel_data, fs=fs, nperseg=nperseg,
        noverlap=noverlap, window='hann', scaling='density')

    mask = (freqs >= fmin) & (freqs <= fmax)
    Sxx  = Sxx[mask, :]

    if Sxx.size == 0:
        return np.zeros((out_h, out_w), dtype=np.uint8)

    if log_power:
        Sxx = 10.0 * np.log10(Sxx + 1e-12)

    lo, hi = Sxx.min(), Sxx.max()
    if hi > lo:
        Sxx = ((Sxx - lo) / (hi - lo) * 255).astype(np.uint8)
    else:
        Sxx = np.zeros_like(Sxx, dtype=np.uint8)

    # flip so low-freq is at bottom (conventional spectrogram orientation)
    Sxx = Sxx[::-1, :]
    img = Image.fromarray(Sxx, mode='L').resize((out_w, out_h), Image.LANCZOS)
    return np.array(img)


def subject_spectrograms(data, channel_idx, fs,
                         epoch_sec=4.0, fmin=1.0, fmax=40.0,
                         win_sec=1.5, overlap=0.75,
                         out_h=64, out_w=64, log_power=True):
    """
    Slice one channel of a subject's recording into fixed-length epochs
    and return a list of spectrogram arrays.
    """
    ch  = data[channel_idx, :]
    ep  = int(epoch_sec * fs)
    n   = len(ch) // ep
    specs = []
    for i in range(max(n, 1)):
        chunk = ch[i*ep:(i+1)*ep] if n > 0 else ch
        specs.append(make_spectrogram(
            chunk, fs, fmin, fmax, win_sec, overlap, out_h, out_w, log_power))
    return specs


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  MANIFOLD  (PCA + competitive reconstruction)
# ═══════════════════════════════════════════════════════════════════════════════

class Manifold:
    def __init__(self):
        self.atoms   = None   # (N, H, W)  uint8 grayscale
        self.labels  = []     # str label per atom
        self.names   = []     # filename per atom
        self.flat    = None   # (N, D) float32
        self.latent  = None   # (N, 2)
        self.mean    = None
        self.pcs     = None   # (2, D)
        self.sigma   = 1.0
        self.groups  = []     # unique group labels in load order

    @property
    def n(self): return 0 if self.atoms is None else len(self.atoms)

    def build(self, atoms, labels, names):
        """atoms: list of HxW uint8 arrays; labels/names: parallel lists."""
        self.atoms  = np.stack(atoms, 0)          # (N, H, W)
        self.labels = list(labels)
        self.names  = list(names)
        self.groups = list(dict.fromkeys(labels))  # preserves order
        N = len(atoms)
        D = atoms[0].size

        self.flat = self.atoms.reshape(N, D).astype(np.float32) / 255.0
        centered  = self.flat - self.flat.mean(0)
        self.mean = self.flat.mean(0)

        if HAS_SKLEARN and N > 300:
            pca_model    = SkPCA(n_components=2, svd_solver='randomized',
                                 random_state=0)
            self.latent  = pca_model.fit_transform(centered)
            self.pcs     = pca_model.components_
        else:
            # Gram-matrix trick: eigen-decompose (N×N) instead of (D×D)
            G     = centered @ centered.T
            evals, evecs = np.linalg.eigh(G)
            idx   = np.argsort(-evals)
            ev1, ev2 = evecs[:, idx[0]], evecs[:, idx[1]]
            pc1 = centered.T @ ev1;  pc1 /= (np.linalg.norm(pc1) + 1e-12)
            pc2 = centered.T @ ev2;  pc2 /= (np.linalg.norm(pc2) + 1e-12)
            self.pcs    = np.stack([pc1, pc2], 0)
            self.latent = centered @ self.pcs.T

        # Length-scale = median nearest-neighbour distance
        d = np.sqrt(((self.latent[:, None] - self.latent[None]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        self.sigma = float(np.median(d.min(1))) + 1e-9

    def weights(self, point, beta):
        d2      = ((self.latent - point) ** 2).sum(1)
        logits  = -beta * d2 / self.sigma ** 2
        logits -= logits.max()
        w = np.exp(logits)
        return w / (w.sum() + 1e-12)

    def reconstruct(self, w):
        return np.clip((w @ self.flat).reshape(self.atoms.shape[1:]) * 255,
                       0, 255).astype(np.uint8)

    def group_weights(self, w):
        """Return dict {group_label: float} summing atom weights by group."""
        out = {g: 0.0 for g in self.groups}
        for i, lbl in enumerate(self.labels):
            out[lbl] += w[i]
        total = sum(out.values()) + 1e-12
        return {k: v / total for k, v in out.items()}

    def separation_score(self):
        """Centroid distance / pooled std (Fisher-like, dimensionless)."""
        if len(self.groups) < 2:
            return 0.0
        g0, g1 = self.groups[0], self.groups[1]
        idx0 = [i for i, l in enumerate(self.labels) if l == g0]
        idx1 = [i for i, l in enumerate(self.labels) if l == g1]
        if not idx0 or not idx1:
            return 0.0
        m0 = self.latent[idx0].mean(0)
        m1 = self.latent[idx1].mean(0)
        s0 = self.latent[idx0].std(0).mean() + 1e-9
        s1 = self.latent[idx1].std(0).mean() + 1e-9
        return float(np.linalg.norm(m1 - m0) / ((s0 + s1) / 2))

    def project_image(self, arr_hw):
        v  = arr_hw.flatten().astype(np.float32) / 255.0 - self.mean
        return v @ self.pcs.T


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

class App:
    def __init__(self, root):
        self.root  = root
        root.title("EEG Koopman Explorer")
        root.geometry("1220x760")
        root.configure(bg=BG)

        self.mfld    = Manifold()
        self.groups  = []          # list of dicts: {label, folder, color}
        self.cursor  = np.array([0.0, 0.0])
        self.beta    = 4.0
        self.ch_idx  = 0
        self.ch_labels = []        # detected channel names
        self.fs      = 250.0
        self._recon_tk   = None
        self._atom_thumbs = []

        # EEG params (shared IntVar / DoubleVar)
        self.v_fs     = tk.DoubleVar(value=250.0)
        self.v_ch     = tk.IntVar(value=0)
        self.v_fmin   = tk.DoubleVar(value=1.0)
        self.v_fmax   = tk.DoubleVar(value=40.0)
        self.v_epoch  = tk.DoubleVar(value=4.0)
        self.v_win    = tk.DoubleVar(value=1.5)
        self.v_res    = tk.IntVar(value=64)
        self.v_log    = tk.BooleanVar(value=True)

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        left   = tk.Frame(self.root, bg=PANEL, width=285)
        center = tk.Frame(self.root, bg=BG)
        right  = tk.Frame(self.root, bg=BG, width=330)
        left.pack(side=tk.LEFT, fill=tk.Y);  left.pack_propagate(False)
        right.pack(side=tk.RIGHT, fill=tk.Y); right.pack_propagate(False)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        def lbl(p, t, fg=DIM, font=("Consolas", 9), **kw):
            return tk.Label(p, text=t, bg=kw.pop('bg', PANEL),
                            fg=fg, font=font, **kw)
        def btn(p, t, cmd, bg='#2a2a3e', **kw):
            return tk.Button(p, text=t, command=cmd, bg=bg, fg=TEXT,
                             font=("Consolas", 10), relief='flat',
                             activebackground='#3a3a52',
                             activeforeground=TEXT, **kw)

        # ── LEFT ──────────────────────────────────────────────────────────────
        lbl(left, "EEG KOOPMAN EXPLORER", fg='#5ee6a8',
            font=("Consolas", 11, "bold")).pack(pady=(14, 8))

        # Groups frame
        lbl(left, "GROUPS  (folder = label)", fg='#5ee6a8').pack(anchor='w', padx=12)
        gf = tk.Frame(left, bg=PANEL); gf.pack(fill=tk.X, padx=12, pady=4)
        btn(gf, "+ Add group", self.add_group).pack(side=tk.LEFT)

        self.group_frame = tk.Frame(left, bg=PANEL)
        self.group_frame.pack(fill=tk.X, padx=12)

        tk.Frame(left, bg='#2a2a3e', height=1).pack(fill=tk.X, padx=12, pady=10)

        # Parameters
        lbl(left, "PARAMETERS", fg='#5ee6a8').pack(anchor='w', padx=12)
        params = [
            ("Channel idx", self.v_ch,    0,   63, 1),
            ("Sample rate (Hz)", self.v_fs, 100, 2000, 1),
            ("Freq min (Hz)", self.v_fmin, 0.5, 20,  0.5),
            ("Freq max (Hz)", self.v_fmax, 5,   100, 1),
            ("Epoch (sec)",  self.v_epoch, 1,   30,  0.5),
            ("Win (sec)",    self.v_win,   0.5, 10,  0.5),
        ]
        for text, var, lo, hi, res in params:
            row = tk.Frame(left, bg=PANEL); row.pack(fill=tk.X, padx=12, pady=1)
            lbl(row, text, bg=PANEL).pack(side=tk.LEFT)
            tk.Spinbox(row, from_=lo, to=hi, increment=res,
                       textvariable=var, width=6,
                       bg='#1a1a28', fg=TEXT, insertbackground=TEXT,
                       relief='flat').pack(side=tk.RIGHT)

        row = tk.Frame(left, bg=PANEL); row.pack(fill=tk.X, padx=12, pady=1)
        lbl(row, "Resolution (px)", bg=PANEL).pack(side=tk.LEFT)
        ttk.Combobox(row, textvariable=self.v_res,
                     values=[32, 48, 64, 96], width=5,
                     state='readonly').pack(side=tk.RIGHT)

        row = tk.Frame(left, bg=PANEL); row.pack(fill=tk.X, padx=12, pady=1)
        lbl(row, "Log power", bg=PANEL).pack(side=tk.LEFT)
        tk.Checkbutton(row, variable=self.v_log, bg=PANEL,
                       selectcolor=PANEL, fg=TEXT,
                       activebackground=PANEL).pack(side=tk.RIGHT)

        lbl(left, "Channel name (if known):").pack(anchor='w', padx=12, pady=(6, 0))
        self.ch_name_var = tk.StringVar()
        tk.Entry(left, textvariable=self.ch_name_var, bg='#1a1a28',
                 fg='#5ee6a8', width=12, relief='flat',
                 font=("Consolas", 9)).pack(anchor='w', padx=12)

        tk.Frame(left, bg='#2a2a3e', height=1).pack(fill=tk.X, padx=12, pady=10)

        btn(left, "▶  PROCESS", self.process,
            bg='#1a4a2a').pack(fill=tk.X, padx=12, pady=4)

        self.progress = ttk.Progressbar(left, mode='determinate')
        self.progress.pack(fill=tk.X, padx=12, pady=2)

        btn(left, "Save dictionary", self.save_dict).pack(fill=tk.X, padx=12, pady=2)
        btn(left, "Load dictionary", self.load_dict).pack(fill=tk.X, padx=12, pady=2)

        self.status = lbl(left, "Add groups to begin.",
                          fg='#5ee6a8', wraplength=250, justify=tk.LEFT)
        self.status.pack(anchor='w', padx=12, pady=8)

        # ── CENTER ────────────────────────────────────────────────────────────
        lbl(center, "KOOPMAN MANIFOLD  (spectrograms in Gabor-frequency space)",
            bg=BG, fg=DIM, font=("Consolas", 10, "bold")).pack(pady=(14, 4))

        self.canvas = tk.Canvas(center, width=CANVAS_W, height=CANVAS_H,
                                bg='#080810', highlightthickness=1,
                                highlightbackground='#2a2a3e')
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._drag)
        self.canvas.bind("<B1-Motion>", self._drag)

        bf = tk.Frame(center, bg=BG); bf.pack(fill=tk.X, padx=20, pady=(6, 0))
        lbl(bf, "competition β", bg=BG, fg=TEXT).pack(side=tk.LEFT)
        self.beta_lbl = lbl(bf, "4.0", bg=BG, fg='#5ee6a8',
                            font=("Consolas", 10, "bold"))
        self.beta_lbl.pack(side=tk.LEFT, padx=8)
        self.beta_scale = tk.Scale(center, from_=0, to=100, orient=tk.HORIZONTAL,
                                   bg=BG, fg=TEXT, highlightthickness=0,
                                   troughcolor='#2a2a3e',
                                   command=self._on_beta)
        self.beta_scale.set(15)
        self.beta_scale.pack(fill=tk.X, padx=20)
        lbl(center, "low → blend across manifold  |  high → snap to nearest",
            bg=BG, fg=DIM, font=("Consolas", 8)).pack()

        # ── RIGHT ─────────────────────────────────────────────────────────────
        lbl(right, "RECONSTRUCTION", bg=BG, fg=DIM,
            font=("Consolas", 11, "bold")).pack(pady=(14, 4))
        self.recon_lbl = tk.Label(right, bg='black',
                                  width=RECON_SZ, height=RECON_SZ)
        self.recon_lbl.pack()

        btn(right, "⊕  Project query image here", self.project_query,
            bg='#2a2a3e').pack(fill=tk.X, padx=16, pady=(8, 4))

        lbl(right, "GROUP WEIGHTS  (active region)", bg=BG,
            fg='#5ee6a8').pack(anchor='w', padx=16, pady=(10, 2))
        self.gw_frame = tk.Frame(right, bg=BG); self.gw_frame.pack(
            fill=tk.X, padx=16)

        lbl(right, "SEPARATION SCORE", bg=BG, fg='#5ee6a8').pack(
            anchor='w', padx=16, pady=(10, 2))
        self.sep_lbl = lbl(right, "—", bg=BG, fg=TEXT,
                           font=("Consolas", 13, "bold"))
        self.sep_lbl.pack(anchor='w', padx=16)
        lbl(right, ">2 = strong separation  |  <1 = overlapping",
            bg=BG, fg=DIM, font=("Consolas", 8)).pack(anchor='w', padx=16)

        self.atoms_lbl = lbl(right, "", bg=BG, fg=DIM,
                             font=("Consolas", 9), justify=tk.LEFT)
        self.atoms_lbl.pack(anchor='w', padx=16, pady=(12, 0))

    # ── GROUPS ────────────────────────────────────────────────────────────────
    def add_group(self):
        folder = filedialog.askdirectory(title="Select EEG folder for one group")
        if not folder:
            return
        label = os.path.basename(folder.rstrip('/\\'))
        color = GROUP_COLORS[len(self.groups) % len(GROUP_COLORS)]
        entry = {'label': label, 'folder': folder, 'color': color}
        self.groups.append(entry)
        self._render_group_list()

    def _render_group_list(self):
        for w in self.group_frame.winfo_children():
            w.destroy()
        for i, g in enumerate(self.groups):
            row = tk.Frame(self.group_frame, bg=PANEL); row.pack(fill=tk.X, pady=2)
            tk.Label(row, bg=g['color'], width=2).pack(side=tk.LEFT, padx=(0, 4))
            tk.Label(row, text=g['label'][:18], bg=PANEL,
                     fg=g['color'], font=("Consolas", 9)).pack(side=tk.LEFT)
            idx = i  # capture
            tk.Button(row, text='✕', bg=PANEL, fg=DIM, relief='flat',
                      command=lambda i=idx: self._remove_group(i)).pack(
                          side=tk.RIGHT)
            n = len(glob.glob(os.path.join(g['folder'], '*.mat')) +
                    glob.glob(os.path.join(g['folder'], '*.edf')) +
                    glob.glob(os.path.join(g['folder'], '*.npy')) +
                    glob.glob(os.path.join(g['folder'], '*.csv')))
            tk.Label(row, text=f"{n} files", bg=PANEL,
                     fg=DIM, font=("Consolas", 9)).pack(side=tk.RIGHT, padx=4)

    def _remove_group(self, i):
        self.groups.pop(i)
        self._render_group_list()

    # ── PROCESSING ────────────────────────────────────────────────────────────
    def process(self):
        if not self.groups:
            messagebox.showinfo("No groups", "Add at least one group folder.")
            return
        threading.Thread(target=self._process_thread, daemon=True).start()

    def _process_thread(self):
        try:
            self._set_status("Scanning files...")
            atoms, labels, names = [], [], []
            ch   = int(self.v_ch.get())
            fs   = float(self.v_fs.get())
            fmin = float(self.v_fmin.get())
            fmax = float(self.v_fmax.get())
            epoch= float(self.v_epoch.get())
            win  = float(self.v_win.get())
            res  = int(self.v_res.get())
            logp = bool(self.v_log.get())
            ch_name = self.ch_name_var.get().strip()

            exts = ('*.mat','*.edf','*.npy','*.csv')
            all_files = []
            for g in self.groups:
                paths = []
                for e in exts:
                    paths += glob.glob(os.path.join(g['folder'], e))
                all_files.append((g['label'], sorted(paths)))

            total = sum(len(p) for _, p in all_files)
            done  = 0
            self.progress['maximum'] = max(total, 1)

            for label, paths in all_files:
                for path in paths:
                    fname = os.path.basename(path)
                    self._set_status(f"[{label}]  {fname}")
                    try:
                        data, file_fs, ch_lbls = load_eeg(path)

                        # Determine channel index
                        cidx = ch
                        if ch_name and ch_lbls:
                            for ci, cl in enumerate(ch_lbls):
                                if cl.strip().lower() == ch_name.lower():
                                    cidx = ci; break
                        if cidx >= data.shape[0]:
                            cidx = 0

                        # Use detected sampling rate if user left default
                        use_fs = file_fs if (file_fs and abs(fs - 250) < 5) else fs

                        specs = subject_spectrograms(
                            data, cidx, use_fs,
                            epoch_sec=epoch, fmin=fmin, fmax=fmax,
                            win_sec=win, overlap=0.75,
                            out_h=res, out_w=res, log_power=logp)

                        for i, s in enumerate(specs):
                            atoms.append(s)
                            labels.append(label)
                            names.append(f"{fname}  ep{i}")
                    except Exception as e:
                        self._set_status(f"  SKIP {fname}: {e}")

                    done += 1
                    self.progress['value'] = done

            if not atoms:
                self._set_status("No spectrograms generated. Check parameters.")
                return

            self._set_status(f"Building manifold  ({len(atoms)} spectrograms)…")
            self.mfld.build(atoms, labels, names)

            score = self.mfld.separation_score()
            self.root.after(0, lambda: self.sep_lbl.config(
                text=f"{score:.2f}",
                fg=('#5ee6a8' if score > 2 else
                    '#f39c12' if score > 1 else '#e74c3c')))

            self.cursor = self.mfld.latent.mean(0).copy()
            self.root.after(0, self._build_thumbs)
            self.root.after(0, self.redraw)
            self.root.after(0, self.update_recon)
            self._set_status(
                f"Done — {len(atoms)} spectrograms, "
                f"{len(self.mfld.groups)} groups, "
                f"separation {score:.2f}")

        except Exception as e:
            self._set_status(f"Error: {e}")
            import traceback; traceback.print_exc()

    def _set_status(self, msg):
        self.root.after(0, lambda: self.status.config(text=msg))

    # ── MANIFOLD DISPLAY ──────────────────────────────────────────────────────
    def _map_fns(self):
        lat = self.mfld.latent
        lo  = lat.min(0); hi = lat.max(0)
        pad = 0.12 * (hi - lo + 1e-9)
        lo -= pad; hi += pad
        M = 36
        def to_px(p):
            x = M + (p[0]-lo[0])/(hi[0]-lo[0]+1e-9) * (CANVAS_W - 2*M)
            y = CANVAS_H - M - (p[1]-lo[1])/(hi[1]-lo[1]+1e-9) * (CANVAS_H - 2*M)
            return x, y
        def to_lat(px, py):
            x = lo[0] + (px-M)/(CANVAS_W-2*M) * (hi[0]-lo[0])
            y = lo[1] + (CANVAS_H-M-py)/(CANVAS_H-2*M) * (hi[1]-lo[1])
            return np.array([x, y])
        return to_px, to_lat

    def _build_thumbs(self):
        self._atom_thumbs = []
        sz = max(12, min(22, 800 // max(self.mfld.n, 1)))
        for i in range(self.mfld.n):
            im = Image.fromarray(self.mfld.atoms[i], mode='L').resize(
                (sz, sz), Image.NEAREST)
            self._atom_thumbs.append((ImageTk.PhotoImage(im), sz))

    def redraw(self):
        self.canvas.delete("all")
        if self.mfld.latent is None:
            return
        to_px, _ = self._map_fns()
        g_color  = {g: GROUP_COLORS[i % len(GROUP_COLORS)]
                    for i, g in enumerate(self.mfld.groups)}

        # Dots (or tiny thumbs if available)
        for i in range(self.mfld.n):
            x, y  = to_px(self.mfld.latent[i])
            color = g_color.get(self.mfld.labels[i], '#888')
            if i < len(self._atom_thumbs):
                tk_img, sz = self._atom_thumbs[i]
                self.canvas.create_image(x, y, image=tk_img)
                self.canvas.create_rectangle(
                    x-sz/2, y-sz/2, x+sz/2, y+sz/2,
                    outline=color, width=1)
            else:
                self.canvas.create_oval(
                    x-NAV_DOT, y-NAV_DOT, x+NAV_DOT, y+NAV_DOT,
                    fill=color, outline='')

        # Legend
        for i, g in enumerate(self.mfld.groups):
            self.canvas.create_rectangle(
                10, 12 + i*18, 22, 24 + i*18,
                fill=g_color[g], outline='')
            self.canvas.create_text(
                28, 18 + i*18, anchor='w', text=g,
                fill=g_color[g], font=('Consolas', 9))

        # Cursor
        cx, cy = to_px(self.cursor)
        self.canvas.create_oval(cx-8, cy-8, cx+8, cy+8,
                                outline='white', width=2)
        self.canvas.create_line(cx-14, cy, cx+14, cy, fill='white')
        self.canvas.create_line(cx, cy-14, cx, cy+14, fill='white')

    def _drag(self, event):
        if self.mfld.latent is None:
            return
        _, to_lat = self._map_fns()
        self.cursor = to_lat(event.x, event.y)
        self.redraw()
        self.update_recon()

    def _on_beta(self, val):
        v = float(val) / 100.0
        self.beta = 0.3 + (v ** 1.6) * 29.7
        self.beta_lbl.config(text=f"{self.beta:.1f}")
        if self.mfld.latent is not None:
            self.update_recon()
            self.redraw()

    # ── RECONSTRUCTION + GROUP WEIGHTS ────────────────────────────────────────
    def update_recon(self):
        if self.mfld.latent is None:
            return
        w    = self.mfld.weights(self.cursor, self.beta)
        img  = self.mfld.reconstruct(w)
        pim  = Image.fromarray(img, mode='L').resize(
            (RECON_SZ, RECON_SZ), Image.NEAREST)
        self._recon_tk = ImageTk.PhotoImage(pim)
        self.recon_lbl.config(image=self._recon_tk)

        # Group weight bars
        for w2 in self.gw_frame.winfo_children():
            w2.destroy()
        gw = self.mfld.group_weights(w)
        g_color = {g: GROUP_COLORS[i % len(GROUP_COLORS)]
                   for i, g in enumerate(self.mfld.groups)}
        for g, frac in gw.items():
            row = tk.Frame(self.gw_frame, bg=BG); row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=f"{g[:14]:14s}", bg=BG,
                     fg=g_color[g], font=("Consolas", 10)).pack(side=tk.LEFT)
            tk.Label(row, text=f"{frac*100:5.1f}%", bg=BG,
                     fg=g_color[g], font=("Consolas", 10, "bold")).pack(
                         side=tk.LEFT, padx=4)
            track = tk.Frame(row, bg='#1a1a28', height=10)
            track.pack(side=tk.LEFT, fill=tk.X, expand=True)
            tk.Frame(track, bg=g_color[g],
                     height=10, width=int(frac * 160)).pack(side=tk.LEFT)

        # Top-4 active atoms
        order = np.argsort(-w)[:4]
        lines = []
        for i in order:
            if w[i] > 0.005:
                lines.append(f"{w[i]*100:5.1f}%  {self.mfld.names[i][:28]}")
        eff = 1.0 / (np.sum(w**2) + 1e-12)
        lines.append(f"\neffective atoms: {eff:.1f}")
        self.atoms_lbl.config(text='\n'.join(lines))

    # ── QUERY IMAGE ───────────────────────────────────────────────────────────
    def project_query(self):
        if self.mfld.pcs is None:
            return
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
        if not path:
            return
        res = self.mfld.atoms.shape[1]
        im  = Image.open(path).convert('L').resize((res, res), Image.LANCZOS)
        arr = np.array(im)
        self.cursor = self.mfld.project_image(arr)
        self.redraw(); self.update_recon()
        self._set_status("Query image projected onto manifold.")

    # ── SAVE / LOAD ───────────────────────────────────────────────────────────
    def save_dict(self):
        if self.mfld.n == 0:
            return
        p = filedialog.asksaveasfilename(defaultextension='.npz',
                                         filetypes=[("Dictionary", "*.npz")])
        if not p:
            return
        np.savez_compressed(
            p,
            atoms  = self.mfld.atoms,
            labels = np.array(self.mfld.labels),
            names  = np.array(self.mfld.names),
            mean   = self.mfld.mean,
            pcs    = self.mfld.pcs,
            latent = self.mfld.latent,
            sigma  = self.mfld.sigma,
            groups = np.array(self.mfld.groups))
        self._set_status(f"Saved {os.path.basename(p)}")

    def load_dict(self):
        p = filedialog.askopenfilename(filetypes=[("Dictionary", "*.npz")])
        if not p:
            return
        d = np.load(p, allow_pickle=True)
        m = self.mfld
        m.atoms   = d['atoms']
        m.labels  = list(d['labels'])
        m.names   = list(d['names'])
        m.mean    = d['mean']
        m.pcs     = d['pcs']
        m.latent  = d['latent']
        m.sigma   = float(d['sigma'])
        m.groups  = list(d['groups'])
        m.flat    = m.atoms.reshape(m.n, -1).astype(np.float32) / 255.0
        self.cursor = m.latent.mean(0).copy()
        score = m.separation_score()
        self.sep_lbl.config(
            text=f"{score:.2f}",
            fg=('#5ee6a8' if score > 2 else
                '#f39c12' if score > 1 else '#e74c3c'))
        self._build_thumbs()
        self.redraw(); self.update_recon()
        self._set_status(f"Loaded {m.n} spectrograms, {len(m.groups)} groups.")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    root = tk.Tk()
    App(root)
    root.mainloop()
