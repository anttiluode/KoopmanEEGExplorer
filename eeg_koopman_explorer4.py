"""
EEG KOOPMAN EXPLORER  v2
========================
Adds convex-hull metrics (spread ratio + fragmentation index) and a
per-channel sweep that ranks all electrodes by how much they separate
the groups.  The centroid-distance score is kept but the hull metrics
are the primary scientific readout for the compact/diffuse geometry
seen in this dataset.

Hull metrics:
  spread_ratio   = sick_hull_area / healthy_hull_area
                   (how much more manifold volume the sick group occupies)
  fragmentation  = fraction of sick epochs that land OUTSIDE the
                   healthy group's convex hull
                   (the "attractor-escape" rate)

FORMATS: .mat (raw or EEGLAB), .edf (via mne or pyedflib), .npy, .csv
DEPS:    numpy  scipy  Pillow  mne (preferred for .edf)
OPTIONAL: scikit-learn  (fast PCA when N > 500)
"""

import os, glob, threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
from scipy import signal as sig
from scipy.io import loadmat
from scipy.spatial import ConvexHull, Delaunay
from PIL import Image, ImageTk

try:
    import mne;          HAS_MNE     = True
except ImportError:      HAS_MNE     = False
try:
    import pyedflib;     HAS_EDF     = True
except ImportError:      HAS_EDF     = False
try:
    from sklearn.decomposition import PCA as SkPCA
    HAS_SKLEARN = True
except ImportError:      HAS_SKLEARN = False

CANVAS_W, CANVAS_H = 580, 520
RECON_SZ           = 260
NAV_DOT            = 5
GROUP_COLORS       = ['#2ecc71','#e74c3c','#3498db','#f39c12',
                      '#9b59b6','#1abc9c','#e67e22','#e91e63']
BG, PANEL, TEXT, DIM = '#0d0d14','#15151f','#e0e0ee','#666677'


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  EEG LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _first_2d(mat_dict):
    cands = {k: v for k, v in mat_dict.items()
             if not k.startswith('_') and isinstance(v, np.ndarray) and v.ndim == 2}
    if not cands:
        return None, None
    key = max(cands, key=lambda k: cands[k].size)
    arr = cands[key]
    return key, (arr if arr.shape[0] <= arr.shape[1] else arr.T)

def _detect_eeglab(mat_dict):
    for root_key in ('EEG','eeg','EEGLAB'):
        if root_key not in mat_dict: continue
        s = mat_dict[root_key]
        try:
            data   = np.array(s['data'][0,0],  dtype=float)
            srate  = float(s['srate'][0,0].flat[0])
            chans  = s['chanlocs'][0,0]
            labels = [str(chans['labels'][0,i][0]) for i in range(chans.size)]
            if data.shape[0] > data.shape[1]: data = data.T
            return data, srate, labels
        except Exception:
            pass
    return None, None, None

def load_eeg(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.mat':
        mat = loadmat(path, squeeze_me=False, struct_as_record=True)
        data, fs, labels = _detect_eeglab(mat)
        if data is not None: return data, fs, labels
        for key in ('data','EEG','eeg','y','eegData','rawData','X'):
            if key in mat:
                arr = np.array(mat[key], dtype=float)
                if arr.ndim == 2:
                    if arr.shape[0] > arr.shape[1]: arr = arr.T
                    return arr, None, None
        _, arr = _first_2d(mat)
        if arr is not None: return arr.astype(float), None, None
        raise ValueError(f"No 2-D EEG array in {os.path.basename(path)}")
    elif ext == '.edf':
        if HAS_MNE:
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            return raw.get_data().astype(float), float(raw.info['sfreq']), raw.ch_names
        elif HAS_EDF:
            f = pyedflib.EdfReader(path)
            fs_ = f.getSampleFrequency(0); lbl = f.getSignalLabels()
            data_ = np.stack([f.readSignal(i) for i in range(f.signals_in_file)])
            f._close()
            return data_.astype(float), float(fs_), lbl
        else:
            raise ImportError("Install mne or pyedflib to read .edf files")
    elif ext == '.npy':
        arr = np.load(path).astype(float)
        if arr.ndim == 1: arr = arr[None,:]
        if arr.shape[0] > arr.shape[1]: arr = arr.T
        return arr, None, None
    elif ext == '.csv':
        arr = np.loadtxt(path, delimiter=',').astype(float)
        if arr.ndim == 1: arr = arr[None,:]
        if arr.shape[0] > arr.shape[1]: arr = arr.T
        return arr, None, None
    else:
        raise ValueError(f"Unsupported format: {ext}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  SPECTROGRAM  (= time-varying Gabor transform)
# ═══════════════════════════════════════════════════════════════════════════════

def make_spectrogram(channel_data, fs, fmin=1., fmax=40.,
                     win_sec=1.5, overlap=0.75, out_h=64, out_w=64, log_power=True):
    nperseg  = int(win_sec * fs)
    noverlap = int(nperseg * overlap)
    freqs, _, Sxx = sig.spectrogram(channel_data, fs=fs, nperseg=nperseg,
                                    noverlap=noverlap, window='hann', scaling='density')
    mask = (freqs >= fmin) & (freqs <= fmax)
    Sxx  = Sxx[mask, :]
    if Sxx.size == 0:
        return np.zeros((out_h, out_w), dtype=np.uint8)
    if log_power:
        Sxx = 10. * np.log10(Sxx + 1e-12)
    lo, hi = Sxx.min(), Sxx.max()
    Sxx = ((Sxx - lo) / (hi - lo + 1e-9) * 255).astype(np.uint8)
    Sxx = Sxx[::-1, :]
    return np.array(Image.fromarray(Sxx,'L').resize((out_w,out_h),Image.LANCZOS))

def subject_spectrograms(data, channel_idx, fs, epoch_sec=4., fmin=1., fmax=40.,
                         win_sec=1.5, overlap=0.75, out_h=64, out_w=64, log_power=True):
    ch = data[channel_idx, :]
    ep = int(epoch_sec * fs)
    n  = len(ch) // ep
    specs = []
    for i in range(max(n,1)):
        chunk = ch[i*ep:(i+1)*ep] if n > 0 else ch
        specs.append(make_spectrogram(chunk, fs, fmin, fmax,
                                      win_sec, overlap, out_h, out_w, log_power))
    return specs


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  MANIFOLD
# ═══════════════════════════════════════════════════════════════════════════════

class Manifold:
    def __init__(self):
        self.atoms  = None   # (N,H,W) uint8
        self.labels = []
        self.names  = []
        self.flat   = None   # (N,D) float32
        self.latent = None   # (N,2)
        self.mean   = None
        self.pcs    = None   # (2,D)
        self.sigma  = 1.
        self.groups = []

    @property
    def n(self): return 0 if self.atoms is None else len(self.atoms)

    def build(self, atoms, labels, names):
        self.atoms  = np.stack(atoms, 0)
        self.labels = list(labels)
        self.names  = list(names)
        self.groups = list(dict.fromkeys(labels))
        N, D = len(atoms), atoms[0].size
        self.flat   = self.atoms.reshape(N,D).astype(np.float32) / 255.
        self.mean   = self.flat.mean(0)
        centered    = self.flat - self.mean
        if HAS_SKLEARN and N > 300:
            m = SkPCA(n_components=2, svd_solver='randomized', random_state=0)
            self.latent = m.fit_transform(centered)
            self.pcs    = m.components_
        else:
            G = centered @ centered.T
            ev, evec = np.linalg.eigh(G)
            idx = np.argsort(-ev)
            pc1 = centered.T @ evec[:,idx[0]]; pc1 /= np.linalg.norm(pc1)+1e-12
            pc2 = centered.T @ evec[:,idx[1]]; pc2 /= np.linalg.norm(pc2)+1e-12
            self.pcs    = np.stack([pc1,pc2],0)
            self.latent = centered @ self.pcs.T
        d = np.sqrt(((self.latent[:,None]-self.latent[None])**2).sum(-1))
        np.fill_diagonal(d, np.inf)
        self.sigma = float(np.median(d.min(1))) + 1e-9

    # ── metrics ───────────────────────────────────────────────────────────────
    def separation_score(self):
        if len(self.groups) < 2: return 0.
        i0 = [i for i,l in enumerate(self.labels) if l==self.groups[0]]
        i1 = [i for i,l in enumerate(self.labels) if l==self.groups[1]]
        if not i0 or not i1: return 0.
        m0, m1 = self.latent[i0].mean(0), self.latent[i1].mean(0)
        s0 = self.latent[i0].std(0).mean() + 1e-9
        s1 = self.latent[i1].std(0).mean() + 1e-9
        return float(np.linalg.norm(m1-m0) / ((s0+s1)/2))

    def hull_metrics(self):
        """
        spread_ratio   : sick_hull_area / healthy_hull_area
        fragmentation  : fraction of sick epochs outside the healthy hull
        Returns None if not enough points or fewer than 2 groups.
        """
        if len(self.groups) < 2: return None
        g = {grp: [i for i,l in enumerate(self.labels) if l==grp]
             for grp in self.groups}
        pts = {grp: self.latent[idx] for grp, idx in g.items()}
        # need at least 3 non-collinear points per group
        for grp, p in pts.items():
            if len(p) < 4: return None
        try:
            hulls = {grp: ConvexHull(p) for grp, p in pts.items()}
        except Exception:
            return None

        g0, g1 = self.groups[0], self.groups[1]   # sick, healthy (load order)
        area0 = hulls[g0].volume   # in 2-D, .volume = polygon area
        area1 = hulls[g1].volume
        spread_ratio = area0 / (area1 + 1e-12)

        # fragmentation: fraction of group-0 epochs outside group-1's hull
        try:
            tri = Delaunay(pts[g1])
            outside = tri.find_simplex(pts[g0]) < 0
            fragmentation = float(outside.mean())
        except Exception:
            fragmentation = float('nan')

        return dict(
            spread_ratio=spread_ratio,
            fragmentation=fragmentation,
            hulls=hulls,
            pts=pts,
            g0=g0, g1=g1,
        )

    # ── reconstruction ────────────────────────────────────────────────────────
    def weights(self, point, beta):
        d2     = ((self.latent - point)**2).sum(1)
        logits = -beta * d2 / self.sigma**2
        logits -= logits.max()
        w = np.exp(logits)
        return w / (w.sum()+1e-12)

    def reconstruct(self, w):
        return np.clip((w @ self.flat).reshape(self.atoms.shape[1:])*255,
                       0,255).astype(np.uint8)

    def group_weights(self, w):
        out = {g: 0. for g in self.groups}
        for i,l in enumerate(self.labels): out[l] += w[i]
        tot = sum(out.values())+1e-12
        return {k: v/tot for k,v in out.items()}

    def project_image(self, arr_hw):
        v = arr_hw.flatten().astype(np.float32)/255. - self.mean
        return v @ self.pcs.T


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  APP
# ═══════════════════════════════════════════════════════════════════════════════

class App:
    def __init__(self, root):
        self.root = root
        root.title("EEG Koopman Explorer  v2")
        root.geometry("1260x780")
        root.configure(bg=BG)

        self.mfld   = Manifold()
        self.groups = []
        self.cursor = np.array([0.,0.])
        self.beta   = 4.
        self.v_fs   = tk.DoubleVar(value=250.)
        self.v_ch   = tk.IntVar(value=0)
        self.v_fmin = tk.DoubleVar(value=1.)
        self.v_fmax = tk.DoubleVar(value=40.)
        self.v_ep   = tk.DoubleVar(value=4.)
        self.v_win  = tk.DoubleVar(value=1.5)
        self.v_res  = tk.IntVar(value=64)
        self.v_log  = tk.BooleanVar(value=True)
        self.ch_name_var = tk.StringVar()
        self._recon_tk   = None
        self._atom_thumbs = []
        self._hull_cache  = None   # last computed hull_metrics result

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        left   = tk.Frame(self.root, bg=PANEL, width=268)
        center = tk.Frame(self.root, bg=BG)
        right  = tk.Frame(self.root, bg=BG, width=346)
        left.pack(side=tk.LEFT, fill=tk.Y);   left.pack_propagate(False)
        right.pack(side=tk.RIGHT, fill=tk.Y); right.pack_propagate(False)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        def L(p, t, fg=DIM, font=("Consolas",9), **kw):
            return tk.Label(p, text=t, bg=kw.pop('bg',PANEL),
                            fg=fg, font=font, **kw)
        def B(p, t, cmd, bg='#2a2a3e', **kw):
            return tk.Button(p, text=t, command=cmd, bg=bg, fg=TEXT,
                             font=("Consolas",10), relief='flat',
                             activebackground='#3a3a52',
                             activeforeground=TEXT, **kw)

        # ── LEFT ──────────────────────────────────────────────────────────────
        L(left,"EEG KOOPMAN EXPLORER  v2",fg='#5ee6a8',
          font=("Consolas",11,"bold")).pack(pady=(12,6))
        L(left,"GROUPS  (folder = label)",fg='#5ee6a8').pack(anchor='w',padx=12)
        B(tk.Frame(left,bg=PANEL).pack(fill=tk.X,padx=12,pady=3) or left,
          "+ Add group", self.add_group).pack(fill=tk.X,padx=12,pady=3)
        self.gframe = tk.Frame(left,bg=PANEL); self.gframe.pack(fill=tk.X,padx=12)
        tk.Frame(left,bg='#2a2a3e',height=1).pack(fill=tk.X,padx=12,pady=8)
        L(left,"PARAMETERS",fg='#5ee6a8').pack(anchor='w',padx=12)

        params = [("Channel idx",self.v_ch,0,63,1),
                  ("Sample rate (Hz)",self.v_fs,50,2000,1),
                  ("Freq min (Hz)",self.v_fmin,0.5,20,.5),
                  ("Freq max (Hz)",self.v_fmax,5,100,1),
                  ("Epoch (sec)",self.v_ep,1,30,.5),
                  ("Win (sec)",self.v_win,.5,10,.5)]
        for txt,var,lo,hi,res in params:
            row=tk.Frame(left,bg=PANEL); row.pack(fill=tk.X,padx=12,pady=1)
            L(row,txt,bg=PANEL).pack(side=tk.LEFT)
            tk.Spinbox(row,from_=lo,to=hi,increment=res,textvariable=var,width=6,
                       bg='#1a1a28',fg=TEXT,insertbackground=TEXT,
                       relief='flat').pack(side=tk.RIGHT)
        row=tk.Frame(left,bg=PANEL); row.pack(fill=tk.X,padx=12,pady=1)
        L(row,"Resolution (px)",bg=PANEL).pack(side=tk.LEFT)
        ttk.Combobox(row,textvariable=self.v_res,values=[32,48,64,96],
                     width=5,state='readonly').pack(side=tk.RIGHT)
        row=tk.Frame(left,bg=PANEL); row.pack(fill=tk.X,padx=12,pady=1)
        L(row,"Log power",bg=PANEL).pack(side=tk.LEFT)
        tk.Checkbutton(row,variable=self.v_log,bg=PANEL,
                       selectcolor=PANEL,fg=TEXT,
                       activebackground=PANEL).pack(side=tk.RIGHT)
        L(left,"Channel name (if known):").pack(anchor='w',padx=12,pady=(6,0))
        tk.Entry(left,textvariable=self.ch_name_var,bg='#1a1a28',fg='#5ee6a8',
                 width=12,relief='flat',font=("Consolas",9)).pack(anchor='w',padx=12)
        tk.Frame(left,bg='#2a2a3e',height=1).pack(fill=tk.X,padx=12,pady=8)
        B(left,"▶  PROCESS",self.process,bg='#1a4a2a').pack(fill=tk.X,padx=12,pady=3)
        B(left,"⟳  SWEEP ALL CHANNELS",self.sweep_channels,
          bg='#1a2a4a').pack(fill=tk.X,padx=12,pady=3)
        B(left,"⚡  TRANSIENT ANALYSIS",self.run_transient_analysis,
          bg='#2a1a3e').pack(fill=tk.X,padx=12,pady=3)
        B(left,"🔬 SUB-SECOND δ-CODING",self.run_subsecond_analysis,
          bg='#1a3a2a').pack(fill=tk.X,padx=12,pady=3)
        self.progress=ttk.Progressbar(left,mode='determinate')
        self.progress.pack(fill=tk.X,padx=12,pady=2)
        B(left,"Save dictionary",self.save_dict).pack(fill=tk.X,padx=12,pady=2)
        B(left,"Load dictionary",self.load_dict).pack(fill=tk.X,padx=12,pady=2)
        self.status=L(left,"Add groups to begin.",fg='#5ee6a8',
                      wraplength=240,justify=tk.LEFT)
        self.status.pack(anchor='w',padx=12,pady=8)

        # ── CENTER ────────────────────────────────────────────────────────────
        L(center,"KOOPMAN MANIFOLD  (spectrograms in Gabor-frequency space)",
          bg=BG,fg=DIM,font=("Consolas",10,"bold")).pack(pady=(12,4))
        self.canvas=tk.Canvas(center,width=CANVAS_W,height=CANVAS_H,
                              bg='#080810',highlightthickness=1,
                              highlightbackground='#2a2a3e')
        self.canvas.pack()
        self.canvas.bind("<Button-1>",self._drag)
        self.canvas.bind("<B1-Motion>",self._drag)
        bf=tk.Frame(center,bg=BG); bf.pack(fill=tk.X,padx=18,pady=(6,0))
        L(bf,"competition β",bg=BG,fg=TEXT).pack(side=tk.LEFT)
        self.beta_lbl=L(bf,"4.0",bg=BG,fg='#5ee6a8',
                        font=("Consolas",10,"bold")); self.beta_lbl.pack(side=tk.LEFT,padx=8)
        self.beta_scale=tk.Scale(center,from_=0,to=100,orient=tk.HORIZONTAL,
                                 bg=BG,fg=TEXT,highlightthickness=0,
                                 troughcolor='#2a2a3e',command=self._on_beta)
        self.beta_scale.set(15); self.beta_scale.pack(fill=tk.X,padx=18)
        L(center,"low → blend across manifold  |  high → snap to nearest",
          bg=BG,fg=DIM,font=("Consolas",8)).pack()

        # ── RIGHT ─────────────────────────────────────────────────────────────
        L(right,"RECONSTRUCTION",bg=BG,fg=DIM,
          font=("Consolas",11,"bold")).pack(pady=(12,4))
        self.recon_lbl=tk.Label(right,bg='black',
                                width=RECON_SZ,height=RECON_SZ)
        self.recon_lbl.pack()
        B(right,"⊕  Project query image here",self.project_query,
          bg='#2a2a3e').pack(fill=tk.X,padx=14,pady=(8,4))

        L(right,"GROUP WEIGHTS  (active region)",bg=BG,
          fg='#5ee6a8').pack(anchor='w',padx=14,pady=(10,2))
        self.gw_frame=tk.Frame(right,bg=BG); self.gw_frame.pack(fill=tk.X,padx=14)

        L(right,"HULL METRICS",bg=BG,fg='#5ee6a8').pack(anchor='w',padx=14,pady=(10,2))
        self.hull_lbl=L(right,"—",bg=BG,fg=TEXT,
                        font=("Consolas",9),justify=tk.LEFT)
        self.hull_lbl.pack(anchor='w',padx=14)

        L(right,"CENTROID SEPARATION",bg=BG,fg='#5ee6a8').pack(anchor='w',padx=14,pady=(8,2))
        self.sep_lbl=L(right,"—",bg=BG,fg=TEXT,font=("Consolas",12,"bold"))
        self.sep_lbl.pack(anchor='w',padx=14)
        L(right,">2 = strong  |  <1 = overlapping",bg=BG,fg=DIM,
          font=("Consolas",8)).pack(anchor='w',padx=14)
        self.atoms_lbl=L(right,"",bg=BG,fg=DIM,font=("Consolas",9),
                         justify=tk.LEFT)
        self.atoms_lbl.pack(anchor='w',padx=14,pady=(10,0))

    # ── GROUPS ────────────────────────────────────────────────────────────────
    def add_group(self):
        folder=filedialog.askdirectory(title="Select EEG folder for one group")
        if not folder: return
        label=os.path.basename(folder.rstrip('/\\'))
        color=GROUP_COLORS[len(self.groups)%len(GROUP_COLORS)]
        self.groups.append({'label':label,'folder':folder,'color':color})
        self._render_groups()

    def _render_groups(self):
        for w in self.gframe.winfo_children(): w.destroy()
        for i,g in enumerate(self.groups):
            row=tk.Frame(self.gframe,bg=PANEL); row.pack(fill=tk.X,pady=2)
            tk.Label(row,bg=g['color'],width=2).pack(side=tk.LEFT,padx=(0,4))
            tk.Label(row,text=g['label'][:16],bg=PANEL,fg=g['color'],
                     font=("Consolas",9)).pack(side=tk.LEFT)
            idx=i
            tk.Button(row,text='✕',bg=PANEL,fg=DIM,relief='flat',
                      command=lambda i=idx:self._rm_group(i)).pack(side=tk.RIGHT)
            exts=('*.mat','*.edf','*.npy','*.csv')
            n=sum(len(glob.glob(os.path.join(g['folder'],e))) for e in exts)
            tk.Label(row,text=f"{n} files",bg=PANEL,fg=DIM,
                     font=("Consolas",9)).pack(side=tk.RIGHT,padx=4)

    def _rm_group(self,i):
        self.groups.pop(i); self._render_groups()

    # ── PROCESS ───────────────────────────────────────────────────────────────
    def _params(self):
        return dict(ch=int(self.v_ch.get()), fs=float(self.v_fs.get()),
                    fmin=float(self.v_fmin.get()), fmax=float(self.v_fmax.get()),
                    epoch=float(self.v_ep.get()), win=float(self.v_win.get()),
                    res=int(self.v_res.get()), logp=bool(self.v_log.get()),
                    ch_name=self.ch_name_var.get().strip())

    def process(self):
        if not self.groups:
            messagebox.showinfo("No groups","Add at least one group folder."); return
        threading.Thread(target=self._process_thread,
                         kwargs=self._params(), daemon=True).start()

    def _process_thread(self, ch, fs, fmin, fmax, epoch, win, res, logp, ch_name,
                        _return=False):
        """If _return=True, return metrics dict instead of updating UI."""
        try:
            atoms,labels,names=[],[],[]
            exts=('*.mat','*.edf','*.npy','*.csv')
            all_files=[(g['label'],sorted(sum([glob.glob(os.path.join(g['folder'],e))
                                               for e in exts],[])))
                       for g in self.groups]
            total=sum(len(p) for _,p in all_files)
            done=0; skips=[]
            self.progress['maximum']=max(total,1)
            for label,paths in all_files:
                for path in paths:
                    fname=os.path.basename(path)
                    if not _return: self._set_status(f"[{label}]  {fname}")
                    try:
                        data,file_fs,ch_lbls=load_eeg(path)
                        cidx=ch
                        if ch_name and ch_lbls:
                            for ci,cl in enumerate(ch_lbls):
                                if cl.strip().lower()==ch_name.lower():
                                    cidx=ci; break
                        if cidx>=data.shape[0]: cidx=0
                        use_fs=file_fs if file_fs else fs
                        specs=subject_spectrograms(data,cidx,use_fs,
                            epoch_sec=epoch,fmin=fmin,fmax=fmax,
                            win_sec=win,overlap=.75,
                            out_h=res,out_w=res,log_power=logp)
                        for i,s in enumerate(specs):
                            atoms.append(s); labels.append(label)
                            names.append(f"{fname} ep{i}")
                    except Exception as e:
                        skips.append(f"{fname}: {e}")
                    done+=1; self.progress['value']=done

            if not atoms:
                msg=f"No spectrograms generated."+('\n'+'\n'.join(skips[:5]) if skips else '')
                if not _return: self._set_status(msg)
                return None

            if not _return: self._set_status(f"Building manifold ({len(atoms)})…")
            self.mfld.build(atoms,labels,names)
            sep   = self.mfld.separation_score()
            hmets = self.mfld.hull_metrics()

            if _return:
                return dict(ch=ch, sep=sep, hmets=hmets, n=len(atoms))

            self._hull_cache=hmets
            self.root.after(0,lambda: self._update_metric_labels(sep,hmets))
            self.cursor=self.mfld.latent.mean(0).copy()
            self.root.after(0,self._build_thumbs)
            self.root.after(0,self.redraw)
            self.root.after(0,self.update_recon)
            skip_note=f"  ({len(skips)} skipped)" if skips else ""
            self._set_status(f"Done — {len(atoms)} spectrograms, "
                             f"{len(self.mfld.groups)} groups, "
                             f"sep {sep:.2f}, spread {hmets['spread_ratio']:.2f}x"
                             f"{skip_note}" if hmets else
                             f"Done — {len(atoms)} spectrograms{skip_note}")
        except Exception as e:
            if not _return: self._set_status(f"Error: {e}")
            import traceback; traceback.print_exc()
            return None

    def _set_status(self, msg):
        self.root.after(0, lambda: self.status.config(text=msg))

    def _update_metric_labels(self, sep, hmets):
        self.sep_lbl.config(text=f"{sep:.2f}",
            fg=('#5ee6a8' if sep>2 else '#f39c12' if sep>1 else '#e74c3c'))
        if hmets:
            sr   = hmets['spread_ratio']
            frag = hmets['fragmentation']
            g0,g1= hmets['g0'], hmets['g1']
            col  = '#5ee6a8' if sr>2 else '#f39c12' if sr>1.5 else '#e74c3c'
            self.hull_lbl.config(fg=col, text=(
                f"spread ratio  {sr:.2f}x\n"
                f"  ({g0} hull area / {g1} hull area)\n"
                f"fragmentation  {frag*100:.1f}%\n"
                f"  (% of {g0} outside {g1} hull)\n\n"
                f"spread ratio > 2 → attractor fragmentation\n"
                f"fragmentation > 30% → meaningful excursions"))
        else:
            self.hull_lbl.config(text="— (need ≥ 4 pts per group)")

    # ── CHANNEL SWEEP ─────────────────────────────────────────────────────────
    def sweep_channels(self):
        if not self.groups:
            messagebox.showinfo("No groups","Add groups first."); return
        # try to detect max channel count from first file
        try:
            first = sorted(sum([glob.glob(os.path.join(g['folder'],'*.edf'))
                                 for g in self.groups],[]))[0]
            raw = mne.io.read_raw_edf(first,preload=False,verbose=False) if HAS_MNE else None
            n_ch = len(raw.ch_names) if raw else 19
            ch_names = raw.ch_names if raw else [str(i) for i in range(n_ch)]
        except Exception:
            n_ch=19; ch_names=[str(i) for i in range(n_ch)]

        win=tk.Toplevel(self.root)
        win.title("Channel sweep"); win.configure(bg=BG)
        win.geometry("480x560")
        tk.Label(win,text="CHANNEL SWEEP  —  spread ratio per electrode",
                 bg=BG,fg='#5ee6a8',font=("Consolas",11,"bold")).pack(pady=10)
        tk.Label(win,text="spread ratio = sick hull area / healthy hull area",
                 bg=BG,fg=DIM,font=("Consolas",9)).pack()
        txt=tk.Text(win,bg='#0a0a12',fg=TEXT,font=("Consolas",10),
                    relief='flat',state='disabled'); txt.pack(fill=tk.BOTH,expand=True,padx=10,pady=10)
        pb=ttk.Progressbar(win,maximum=n_ch,mode='determinate'); pb.pack(fill=tk.X,padx=10,pady=(0,10))

        def run():
            results=[]
            p=self._params()
            for ci in range(n_ch):
                pp={**p,'ch':ci,'ch_name':''}
                self.root.after(0,lambda ci=ci: pb.__setitem__('value',ci))
                r=self._process_thread(**pp,_return=True)
                if r and r['hmets']:
                    results.append((ci,ch_names[ci] if ci<len(ch_names) else str(ci),
                                    r['hmets']['spread_ratio'],
                                    r['hmets']['fragmentation']))
                else:
                    results.append((ci,ch_names[ci] if ci<len(ch_names) else str(ci),0.,0.))
            results.sort(key=lambda x:-x[2])
            lines=["idx  name     spread   frag%\n"+"─"*36]
            for idx,name,sr,fr in results:
                lines.append(f" {idx:2d}  {name:<8s}  {sr:6.2f}x  {fr*100:5.1f}%")
            def show():
                txt.config(state='normal'); txt.delete('1.0','end')
                txt.insert('end','\n'.join(lines)); txt.config(state='disabled')
                pb['value']=n_ch
            self.root.after(0,show)
        threading.Thread(target=run,daemon=True).start()

    # ── MANIFOLD DISPLAY ──────────────────────────────────────────────────────
    def _map_fns(self):
        lat=self.mfld.latent
        lo=lat.min(0); hi=lat.max(0)
        pad=0.12*(hi-lo+1e-9); lo-=pad; hi+=pad
        M=36
        def tp(p):
            return (M+(p[0]-lo[0])/(hi[0]-lo[0]+1e-9)*(CANVAS_W-2*M),
                    CANVAS_H-M-(p[1]-lo[1])/(hi[1]-lo[1]+1e-9)*(CANVAS_H-2*M))
        def tl(px,py):
            return np.array([lo[0]+(px-M)/(CANVAS_W-2*M)*(hi[0]-lo[0]),
                              lo[1]+(CANVAS_H-M-py)/(CANVAS_H-2*M)*(hi[1]-lo[1])])
        return tp,tl

    def _build_thumbs(self):
        self._atom_thumbs=[]
        if self.mfld.n>80: return
        sz=max(12,min(22,800//max(self.mfld.n,1)))
        for i in range(self.mfld.n):
            im=Image.fromarray(self.mfld.atoms[i],'L').resize((sz,sz),Image.NEAREST)
            self._atom_thumbs.append((ImageTk.PhotoImage(im),sz))

    def redraw(self):
        self.canvas.delete("all")
        if self.mfld.latent is None: return
        tp,_=self._map_fns()
        gc={g:GROUP_COLORS[i%len(GROUP_COLORS)] for i,g in enumerate(self.mfld.groups)}

        # ── convex hull outlines ───────────────────────────────────────────────
        if self._hull_cache:
            hmets=self._hull_cache
            for grp,hull in hmets['hulls'].items():
                pts_px=[tp(hmets['pts'][grp][v]) for v in hull.vertices]
                flat=[c for p in pts_px for c in p]
                if len(flat)>=6:
                    self.canvas.create_polygon(*flat,outline=gc.get(grp,'#888'),
                                               fill='',width=1,dash=(4,3))

        # ── dots ──────────────────────────────────────────────────────────────
        for i in range(self.mfld.n):
            x,y=tp(self.mfld.latent[i]); col=gc.get(self.mfld.labels[i],'#888')
            if i<len(self._atom_thumbs):
                tk_img,sz=self._atom_thumbs[i]
                self.canvas.create_image(x,y,image=tk_img)
                self.canvas.create_rectangle(x-sz/2,y-sz/2,x+sz/2,y+sz/2,
                                             outline=col,width=1)
            else:
                self.canvas.create_oval(x-NAV_DOT,y-NAV_DOT,x+NAV_DOT,y+NAV_DOT,
                                        fill=col,outline='')

        # ── legend ────────────────────────────────────────────────────────────
        for i,g in enumerate(self.mfld.groups):
            self.canvas.create_rectangle(10,12+i*18,22,24+i*18,fill=gc[g],outline='')
            self.canvas.create_text(28,18+i*18,anchor='w',text=g,
                                    fill=gc[g],font=('Consolas',9))

        # ── cursor ────────────────────────────────────────────────────────────
        cx,cy=tp(self.cursor)
        self.canvas.create_oval(cx-8,cy-8,cx+8,cy+8,outline='white',width=2)
        self.canvas.create_line(cx-14,cy,cx+14,cy,fill='white')
        self.canvas.create_line(cx,cy-14,cx,cy+14,fill='white')

    def _drag(self,event):
        if self.mfld.latent is None: return
        _,tl=self._map_fns()
        self.cursor=tl(event.x,event.y)
        self.redraw(); self.update_recon()

    def _on_beta(self,val):
        v=float(val)/100.; self.beta=0.3+(v**1.6)*29.7
        self.beta_lbl.config(text=f"{self.beta:.1f}")
        if self.mfld.latent is not None:
            self.update_recon(); self.redraw()

    # ── RECONSTRUCTION ────────────────────────────────────────────────────────
    def update_recon(self):
        if self.mfld.latent is None: return
        w  =self.mfld.weights(self.cursor,self.beta)
        img=self.mfld.reconstruct(w)
        pim=Image.fromarray(img,'L').resize((RECON_SZ,RECON_SZ),Image.NEAREST)
        self._recon_tk=ImageTk.PhotoImage(pim)
        self.recon_lbl.config(image=self._recon_tk)
        for ww in self.gw_frame.winfo_children(): ww.destroy()
        gw=self.mfld.group_weights(w)
        gc={g:GROUP_COLORS[i%len(GROUP_COLORS)] for i,g in enumerate(self.mfld.groups)}
        for g,frac in gw.items():
            row=tk.Frame(self.gw_frame,bg=BG); row.pack(fill=tk.X,pady=1)
            tk.Label(row,text=f"{g[:12]:12s}",bg=BG,fg=gc[g],
                     font=("Consolas",10)).pack(side=tk.LEFT)
            tk.Label(row,text=f"{frac*100:5.1f}%",bg=BG,fg=gc[g],
                     font=("Consolas",10,"bold")).pack(side=tk.LEFT,padx=4)
            tr=tk.Frame(row,bg='#1a1a28',height=10); tr.pack(side=tk.LEFT,fill=tk.X,expand=True)
            tk.Frame(tr,bg=gc[g],height=10,
                     width=int(frac*150)).pack(side=tk.LEFT)
        order=np.argsort(-w)[:4]
        lines=[]
        for i in order:
            if w[i]>0.005: lines.append(f"{w[i]*100:5.1f}%  {self.mfld.names[i][:30]}")
        eff=1./(np.sum(w**2)+1e-12)
        lines.append(f"\neffective atoms: {eff:.1f}")
        self.atoms_lbl.config(text='\n'.join(lines))

    # ── QUERY / SAVE / LOAD ───────────────────────────────────────────────────
    def project_query(self):
        if self.mfld.pcs is None: return
        path=filedialog.askopenfilename(
            filetypes=[("Images","*.png *.jpg *.jpeg *.bmp")])
        if not path: return
        res=self.mfld.atoms.shape[1]
        im=Image.open(path).convert('L').resize((res,res),Image.LANCZOS)
        self.cursor=self.mfld.project_image(np.array(im))
        self.redraw(); self.update_recon()
        self._set_status("Query image projected onto manifold.")

    def save_dict(self):
        if self.mfld.n==0: return
        p=filedialog.asksaveasfilename(defaultextension='.npz',
                                        filetypes=[("Dictionary","*.npz")])
        if not p: return
        np.savez_compressed(p,atoms=self.mfld.atoms,
                            labels=np.array(self.mfld.labels),
                            names=np.array(self.mfld.names),
                            mean=self.mfld.mean,pcs=self.mfld.pcs,
                            latent=self.mfld.latent,sigma=self.mfld.sigma,
                            groups=np.array(self.mfld.groups))
        self._set_status(f"Saved {os.path.basename(p)}")

    def load_dict(self):
        p=filedialog.askopenfilename(filetypes=[("Dictionary","*.npz")])
        if not p: return
        d=np.load(p,allow_pickle=True); m=self.mfld
        m.atoms=d['atoms']; m.labels=list(d['labels']); m.names=list(d['names'])
        m.mean=d['mean']; m.pcs=d['pcs']; m.latent=d['latent']
        m.sigma=float(d['sigma']); m.groups=list(d['groups'])
        m.flat=m.atoms.reshape(m.n,-1).astype(np.float32)/255.
        self.cursor=m.latent.mean(0).copy()
        sep=m.separation_score(); hmets=m.hull_metrics()
        self._hull_cache=hmets
        self._update_metric_labels(sep,hmets)
        self._build_thumbs(); self.redraw(); self.update_recon()
        self._set_status(f"Loaded {m.n} spectrograms, {len(m.groups)} groups.")


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSIENT ANALYZER
# Tests the delta-coding prediction:
#   If the spike IS the derivative of the standing wave, then burst→silence
#   transitions in spectral entropy should land on known manifold attractors
#   (low manifold distance) in healthy subjects.  In schizophrenia the system
#   collapses onto idiosyncratic internal states — the correlation breaks.
#
# Method:
#   Each stored epoch (atom) has a spectral entropy.  High entropy = diffuse /
#   exploratory state; low entropy = locked / stable attractor.  Within each
#   subject's epoch sequence (temporal order by ep-number) we detect
#   high→low transitions (burst→silence) and measure how far the post-silence
#   epoch sits from the manifold's nearest known attractor.
#
#   Prediction:
#     healthy:       silence_entropy ↓  ↔  manifold_distance ↓   (r < 0, p < .05)
#     schizophrenia: correlation absent or reversed
#     (locked epochs land on idiosyncratic states, not the group's attractor)
# ═══════════════════════════════════════════════════════════════════════════════

class TransientAnalyzer:
    def __init__(self, mfld):
        self.mfld = mfld

    def spectral_entropy(self, atom):
        """Shannon entropy of a flattened spectrogram. Higher = more diffuse."""
        p = atom.flatten().astype(np.float64) + 1e-9
        p /= p.sum()
        return float(-np.sum(p * np.log(p)))

    def subject_sequences(self):
        """Group atoms by subject (everything before ' ep' in the name),
        sorted by epoch index."""
        out = {}
        for i, (name, label) in enumerate(zip(self.mfld.names, self.mfld.labels)):
            subj = name.split(' ep')[0] if ' ep' in name else name
            if subj not in out:
                out[subj] = {'label': label, 'epochs': []}
            ep_num = i
            if ' ep' in name:
                try:    ep_num = int(name.split(' ep')[1].strip())
                except: ep_num = i
            out[subj]['epochs'].append({'idx': i, 'ep': ep_num,
                                        'entropy': self.spectral_entropy(self.mfld.atoms[i])})
        for s in out:
            out[s]['epochs'].sort(key=lambda e: e['ep'])
        return out

    def find_transitions(self, entropies, k=3):
        """Return high→low transition events in an epoch-ordered entropy sequence.
        Each event: pre-mean (burst), post-mean (silence), magnitude (delta)."""
        e = np.array(entropies)
        n = len(e)
        if n < 2*k + 2:
            return []
        med, std = np.median(e), np.std(e) + 1e-9
        hi = med + 0.25*std
        lo = med - 0.25*std
        events = []
        for t in range(k, n - k):
            pre  = float(np.mean(e[t-k:t]))
            post = float(np.mean(e[t:t+k]))
            if pre > hi and post < lo:
                events.append({'t': t, 'pre': pre, 'post': post,
                                'delta': pre - post})
        return events

    def nearest_dist(self, idx):
        """Euclidean distance in latent space from atom idx to nearest neighbour."""
        d = np.sqrt(((self.mfld.latent - self.mfld.latent[idx])**2).sum(1))
        d[idx] = np.inf
        return float(d.min())

    def run(self, status_cb=None):
        """Run full analysis. Returns (points_list, per_group_stats_dict)."""
        subjects = self.subject_sequences()
        points = []
        for subj, info in subjects.items():
            epochs = info['epochs']
            if len(epochs) < 8:
                continue
            if status_cb:
                status_cb(f"Transients: {subj} ({len(epochs)} epochs)")
            es = [ep['entropy'] for ep in epochs]
            for ev in self.find_transitions(es):
                # post-transition epoch = first epoch in the 'silence' window
                si = min(ev['t'], len(epochs)-1)
                ep = epochs[si]
                dist = self.nearest_dist(ep['idx'])
                points.append({
                    'subject': subj,
                    'label': info['label'],
                    'burst_entropy':   ev['pre'],
                    'silence_entropy': ev['post'],
                    'delta':           ev['delta'],
                    'manifold_dist':   dist,
                    'atom_idx':        ep['idx'],
                })
        # Per-group Pearson r (silence_entropy vs manifold_dist)
        stats = {}
        try:
            from scipy.stats import pearsonr
        except ImportError:
            pearsonr = None
        for grp in self.mfld.groups:
            pts = [p for p in points if p['label'] == grp]
            if len(pts) < 4 or pearsonr is None:
                stats[grp] = {'r': 0.0, 'p': 1.0, 'n': len(pts)}
                continue
            x = np.array([p['silence_entropy'] for p in pts])
            y = np.array([p['manifold_dist']   for p in pts])
            try:
                r, pv = pearsonr(x, y)
            except Exception:
                r, pv = 0.0, 1.0
            stats[grp] = {'r': float(r), 'p': float(pv), 'n': len(pts)}
        return points, stats


# ── App method (added below load_dict inside the class, but kept here as a
#    module-level function that is injected via monkey-patch at import time) ──

def _run_transient_analysis(self):
    """Open transient-analysis window (calls TransientAnalyzer)."""
    if self.mfld.latent is None:
        messagebox.showinfo("No manifold", "Build a manifold first.")
        return
    win = tk.Toplevel(self.root)
    win.title("Transient Analysis  —  δ-coding prediction")
    win.configure(bg=BG)
    win.geometry("860x660")
    tk.Label(win,
        text="TRANSIENT ANALYSIS  ·  burst → silence → manifold proximity",
        bg=BG, fg='#5ee6a8', font=("Consolas",11,"bold")).pack(pady=(14,2))
    tk.Label(win,
        text="Prediction: silence_entropy ↓  ↔  manifold_distance ↓  (healthy only)\n"
             "If sick subjects lock onto idiosyncratic states the correlation breaks.",
        bg=BG, fg=DIM, font=("Consolas",9)).pack()
    pb = ttk.Progressbar(win, mode='indeterminate')
    pb.pack(fill=tk.X, padx=16, pady=6)
    pb.start(12)
    frame = tk.Frame(win, bg=BG)
    frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)
    stat_lbl = tk.Label(win, text="", bg=BG, fg='#5ee6a8',
                        font=("Consolas",10))
    stat_lbl.pack(pady=4)

    def run():
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

            analyzer = TransientAnalyzer(self.mfld)
            points, stats = analyzer.run(status_cb=self._set_status)

            if not points:
                self.root.after(0, lambda: [
                    pb.stop(),
                    stat_lbl.config(text="No transitions detected. "
                                         "Try shorter epoch length or more subjects.")])
                return

            gc = {g: GROUP_COLORS[i % len(GROUP_COLORS)]
                  for i, g in enumerate(self.mfld.groups)}

            fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
            fig.patch.set_facecolor('#0d0d14')

            # ── left: scatter silence_entropy vs manifold_dist ──────────────
            ax = axes[0]; ax.set_facecolor('#12121e')
            for grp in self.mfld.groups:
                pts = [p for p in points if p['label'] == grp]
                if not pts: continue
                x = np.array([p['silence_entropy'] for p in pts])
                y = np.array([p['manifold_dist']    for p in pts])
                s = stats[grp]
                pstr = '<.001' if s['p']<.001 else f"{s['p']:.3f}"
                lbl = (f"{grp}  r={s['r']:.2f}  p={pstr}  n={s['n']}")
                ax.scatter(x, y, c=gc[grp], s=20, alpha=0.55, label=lbl)
                if len(x) > 2:
                    m, b = np.polyfit(x, y, 1)
                    xr = np.linspace(x.min(), x.max(), 60)
                    ax.plot(xr, m*xr+b, color=gc[grp], lw=2, alpha=0.85)
            ax.set_xlabel("silence entropy (post-transition)", color='#888', fontsize=9)
            ax.set_ylabel("manifold distance (nearest atom)", color='#888', fontsize=9)
            ax.set_title("δ-coding: does locking reach a known attractor?",
                         color='#ddd', fontsize=10)
            ax.tick_params(colors='#666', labelsize=8)
            ax.legend(facecolor='#1a1a28', edgecolor='#333',
                      labelcolor='white', fontsize=7.5, loc='upper left')
            for sp in ax.spines.values(): sp.set_color('#2a2a3e')

            # ── right: correlation bar chart ────────────────────────────────
            ax2 = axes[1]; ax2.set_facecolor('#12121e')
            grps = list(self.mfld.groups)
            rs   = [stats[g]['r'] for g in grps]
            bars = ax2.bar(grps, rs, color=[gc[g] for g in grps],
                           alpha=0.85, width=0.5)
            ax2.axhline(0, color='#444', lw=1)
            ax2.axhspan(-1, -0.3, alpha=0.06, color='#2ecc71',
                        label='strong negative (prediction)')
            ax2.set_ylim(-1, 1)
            ax2.set_ylabel("Pearson r", color='#888', fontsize=9)
            ax2.set_title("Prediction: r < 0 in healthy\n(locked = near attractor)",
                         color='#ddd', fontsize=10)
            ax2.tick_params(colors='#666', labelsize=9)
            ax2.legend(facecolor='#1a1a28', edgecolor='#333',
                       labelcolor='white', fontsize=7.5)
            for i, (g, bar) in enumerate(zip(grps, bars)):
                s = stats[g]
                sig = '**' if s['p']<.01 else ('*' if s['p']<.05 else 'n.s.')
                yoff = 0.04 if bar.get_height() >= 0 else -0.07
                ax2.text(bar.get_x()+bar.get_width()/2,
                         bar.get_height()+yoff,
                         f"p={s['p']:.3f}\n{sig}",
                         ha='center', va='bottom', color='white', fontsize=8)
            for sp in ax2.spines.values(): sp.set_color('#2a2a3e')

            fig.tight_layout(pad=2.2)

            summary = "  |  ".join(
                f"{g}: r={stats[g]['r']:+.3f}  "
                f"({'**' if stats[g]['p']<.01 else '*' if stats[g]['p']<.05 else 'n.s.'})"
                f"  n={stats[g]['n']}"
                for g in grps)

            def show():
                pb.stop(); pb.pack_forget()
                canvas = FigureCanvasTkAgg(fig, master=frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                stat_lbl.config(text=summary)
                self._set_status(
                    f"Transients: {len(points)} transitions  |  " + summary)
            self.root.after(0, show)

        except Exception as e:
            import traceback; traceback.print_exc()
            self.root.after(0, lambda: [pb.stop(),
                                        stat_lbl.config(text=f"Error: {e}")])

    threading.Thread(target=run, daemon=True).start()


# Attach the method to App at module level so it works with the class defined above
App.run_transient_analysis = _run_transient_analysis


# ═══════════════════════════════════════════════════════════════════════════════
# SUB-SECOND ANALYZER  (the correct timescale for the δ-coding prediction)
#
# The epoch-level transient test was null because 4-second epochs average over
# dozens of neural cycles, hiding the within-epoch dynamics.
#
# This implementation operates at 100 ms resolution on the raw EEG:
#   1. Bandpass filter into broadband (80-120 Hz) and alpha (8-13 Hz).
#   2. Compute RMS power in 100 ms sliding windows.
#   3. Detect burst→silence pairs:
#        burst  = broadband > mean + 1.5 SD
#        silence = alpha < mean - 1.0 SD  within 200-500 ms of burst end.
#   4. For each silence event, extract a 4-second spectrogram centered on it
#      (same parameters as the manifold atoms — apples to apples).
#   5. Project that spectrogram into the Koopman manifold.
#   6. Correlate: silence_depth (alpha suppression in SD) vs manifold_distance.
#
# Permutation test: shuffle group labels, recompute |Δr|, report p(Δr ≥ obs).
# ═══════════════════════════════════════════════════════════════════════════════

class SubSecondAnalyzer:
    def __init__(self, mfld, groups, params):
        self.mfld   = mfld
        self.groups = groups   # list of {label, folder, color}
        self.p      = params   # dict from App._params()

    # ── signal utilities ──────────────────────────────────────────────────────
    @staticmethod
    def bandpass(x, lo, hi, fs):
        from scipy.signal import butter, filtfilt
        nyq = fs / 2.0
        hi  = min(hi, nyq * 0.98)
        lo  = max(lo, nyq * 0.005)
        if lo >= hi:
            return np.zeros_like(x)
        b, a = butter(4, [lo/nyq, hi/nyq], btype='band')
        pad  = min(len(x)-1, 3*max(len(b), len(a)))
        return filtfilt(b, a, x, padlen=pad)

    @staticmethod
    def sliding_rms(x, win):
        """Vectorised sliding RMS via cumulative sum of squares."""
        x2 = x.astype(np.float64) ** 2
        cs = np.concatenate([[0.0], np.cumsum(x2)])
        hw = win // 2
        n  = len(x)
        s  = np.maximum(0, np.arange(n) - hw)
        e  = np.minimum(n, np.arange(n) + hw + 1)
        return np.sqrt((cs[e] - cs[s]) / (e - s))

    # ── event detection ───────────────────────────────────────────────────────
    @staticmethod
    def detect_events(bb_pow, al_pow, fs, burst_z=1.5, silence_z=1.0,
                      gap_min_ms=200, gap_max_ms=500, refractory_s=1.0):
        bb_m, bb_s = bb_pow.mean(), bb_pow.std() + 1e-9
        al_m, al_s = al_pow.mean(), al_pow.std() + 1e-9
        bt  = bb_m + burst_z  * bb_s
        st  = al_m - silence_z * al_s
        glo = int(gap_min_ms * fs / 1000)
        ghi = int(gap_max_ms * fs / 1000)
        ref = int(refractory_s * fs)
        n   = len(bb_pow)
        events, last, t = [], -ref, 0
        while t < n - ghi:
            if bb_pow[t] > bt:
                t0 = t
                while t < n and bb_pow[t] > bt:
                    t += 1
                t1 = t      # burst end
                amp = (bb_pow[t0:t1].max() - bb_m) / bb_s
                if t1 > last + ref:
                    s_start = t1 + glo
                    s_end   = min(t1 + ghi, n)
                    if s_start < s_end:
                        win = al_pow[s_start:s_end]
                        mi  = int(win.argmin())
                        if win[mi] < st:
                            samp = s_start + mi
                            depth = (al_m - win[mi]) / al_s
                            events.append({'burst_t': t0, 'silence_t': samp,
                                           'burst_amp': float(amp),
                                           'silence_depth': float(depth)})
                            last = samp
                            t = samp + ref
                            continue
            t += 1
        return events

    # ── main run ──────────────────────────────────────────────────────────────
    def run(self, status_cb=None):
        from scipy.stats import pearsonr
        exts = ('*.mat','*.edf','*.npy','*.csv')
        ch   = self.p['ch'];  ch_name = self.p.get('ch_name','')
        fs_u = float(self.p['fs'])
        res  = self.mfld.atoms.shape[1]
        points = []

        for grp in self.groups:
            label, folder = grp['label'], grp['folder']
            paths = sorted(sum([glob.glob(os.path.join(folder,e)) for e in exts],[]))
            for path in paths:
                fname = os.path.basename(path)
                if status_cb:
                    status_cb(f"[{label}]  {fname}")
                try:
                    data, file_fs, ch_lbls = load_eeg(path)
                    cidx = ch
                    if ch_name and ch_lbls:
                        for ci, cl in enumerate(ch_lbls):
                            if cl.strip().lower() == ch_name.lower():
                                cidx = ci; break
                    if cidx >= data.shape[0]:
                        cidx = 0
                    fs  = float(file_fs) if file_fs else fs_u
                    raw = data[cidx, :]
                    if len(raw) < 10 * fs:
                        continue

                    # Filter
                    hi_bb = min(120.0, fs/2 - 2)
                    bb = self.bandpass(raw, 80.0,  hi_bb, fs)
                    al = self.bandpass(raw,  8.0,  13.0,  fs)

                    # Sliding RMS at 100 ms
                    w100 = max(4, int(0.10 * fs))
                    bb_p = self.sliding_rms(bb, w100)
                    al_p = self.sliding_rms(al, w100)

                    # Events
                    events = self.detect_events(bb_p, al_p, fs)

                    # For each event: extract 4-s spectrogram → project → distance
                    half_ep = int(2.0 * fs)
                    for ev in events:
                        ct = ev['silence_t']
                        if ct - half_ep < 0 or ct + half_ep > len(raw):
                            continue
                        chunk = raw[ct-half_ep : ct+half_ep]
                        spec  = make_spectrogram(
                            chunk, fs,
                            fmin=self.p['fmin'], fmax=self.p['fmax'],
                            win_sec=self.p['win'], overlap=0.75,
                            out_h=res, out_w=res, log_power=True)
                        latent = self.mfld.project_image(spec)
                        d  = np.sqrt(((self.mfld.latent - latent)**2).sum(1))
                        points.append({
                            'label':         label,
                            'subject':       fname,
                            'silence_depth': ev['silence_depth'],
                            'burst_amp':     ev['burst_amp'],
                            'manifold_dist': float(d.min()),
                        })
                except Exception as e:
                    if status_cb:
                        status_cb(f"  skip {fname}: {e}")

        # Per-group Pearson r
        stats = {}
        for grp in self.mfld.groups:
            pts = [p for p in points if p['label'] == grp]
            if len(pts) < 4:
                stats[grp] = {'r': 0.0, 'p': 1.0, 'n': len(pts)}; continue
            x = np.array([p['silence_depth'] for p in pts])
            y = np.array([p['manifold_dist']  for p in pts])
            try:    r, pv = pearsonr(x, y)
            except: r, pv = 0.0, 1.0
            stats[grp] = {'r': float(r), 'p': float(pv), 'n': len(pts)}

        perm_p = self._permutation_p(points)
        return points, stats, perm_p

    def _permutation_p(self, points, n_perm=1000):
        from scipy.stats import pearsonr
        if len(self.mfld.groups) < 2 or len(points) < 8:
            return 1.0
        g0, g1 = self.mfld.groups[0], self.mfld.groups[1]
        labs   = np.array([p['label']         for p in points])
        x_all  = np.array([p['silence_depth'] for p in points])
        y_all  = np.array([p['manifold_dist']  for p in points])

        def delta_r(ls):
            i0 = ls == g0;  i1 = ls == g1
            r0 = pearsonr(x_all[i0], y_all[i0])[0] if i0.sum() > 3 else 0.0
            r1 = pearsonr(x_all[i1], y_all[i1])[0] if i1.sum() > 3 else 0.0
            return abs(r0 - r1)

        obs = delta_r(labs)
        rng = np.random.default_rng(42)
        hits = sum(delta_r(rng.permutation(labs)) >= obs for _ in range(n_perm))
        return (hits + 1) / (n_perm + 1)


# ── App method ────────────────────────────────────────────────────────────────

def _run_subsecond_analysis(self):
    """Launch sub-second (100 ms window) burst→silence δ-coding test."""
    if self.mfld.latent is None:
        messagebox.showinfo("No manifold", "Build a manifold first."); return
    if not self.groups:
        messagebox.showinfo("No groups", "Groups needed to reload raw EEG."); return

    win = tk.Toplevel(self.root)
    win.title("Sub-second δ-coding  —  100 ms windows")
    win.configure(bg=BG); win.geometry("900x700")

    tk.Label(win,
        text="SUB-SECOND  ·  100 ms windows  ·  broadband burst → alpha silence",
        bg=BG, fg='#5ee6a8', font=("Consolas",11,"bold")).pack(pady=(14,2))
    tk.Label(win,
        text="burst: 80-120 Hz > +1.5 SD  →  silence: alpha 8-13 Hz < -1 SD  (200-500 ms later)\n"
             "projection: 4-s spectrogram at silence time → Koopman manifold distance\n"
             "prediction: deeper silence → nearer attractor  (healthy)  |  broken in sick",
        bg=BG, fg=DIM, font=("Consolas",9)).pack()

    pb = ttk.Progressbar(win, mode='indeterminate')
    pb.pack(fill=tk.X, padx=16, pady=6); pb.start(12)
    frame = tk.Frame(win, bg=BG); frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
    stat_lbl = tk.Label(win, text="", bg=BG, fg='#5ee6a8', font=("Consolas",10))
    stat_lbl.pack(pady=4)

    def run():
        try:
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

            analyzer = SubSecondAnalyzer(self.mfld, self.groups, self._params())
            points, stats, perm_p = analyzer.run(status_cb=self._set_status)

            if not points:
                self.root.after(0, lambda: [pb.stop(),
                    stat_lbl.config(text="No events detected. Check channel / sample rate / broadband available.")])
                return

            gc = {g: GROUP_COLORS[i % len(GROUP_COLORS)]
                  for i, g in enumerate(self.mfld.groups)}
            fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))
            fig.patch.set_facecolor('#0d0d14')

            # ── scatter ───────────────────────────────────────────────────────
            ax = axes[0]; ax.set_facecolor('#12121e')
            for grp in self.mfld.groups:
                pts = [p for p in points if p['label'] == grp]
                if not pts: continue
                x  = np.array([p['silence_depth'] for p in pts])
                y  = np.array([p['manifold_dist']  for p in pts])
                s  = stats[grp]
                pstr = '<.001' if s['p']<.001 else f"{s['p']:.3f}"
                ax.scatter(x, y, c=gc[grp], s=12, alpha=0.40,
                           label=f"{grp}  r={s['r']:.3f}  p={pstr}  n={s['n']}")
                if len(x) > 3:
                    m, b = np.polyfit(x, y, 1)
                    xr = np.linspace(x.min(), x.max(), 80)
                    ax.plot(xr, m*xr+b, color=gc[grp], lw=2.0, alpha=0.9)
            ax.set_xlabel("silence depth (alpha suppression, SD units)", color='#888', fontsize=9)
            ax.set_ylabel("manifold distance (nearest atom)",            color='#888', fontsize=9)
            ax.set_title("δ-coding at 100 ms:\ndeeper silence → near attractor?",
                         color='#ddd', fontsize=10)
            ax.tick_params(colors='#666', labelsize=8)
            ax.legend(facecolor='#1a1a28', edgecolor='#333', labelcolor='white',
                      fontsize=7.5, loc='upper right')
            for sp in ax.spines.values(): sp.set_color('#2a2a3e')

            # ── correlation bar + permutation ─────────────────────────────────
            ax2 = axes[1]; ax2.set_facecolor('#12121e')
            grps = list(self.mfld.groups)
            rs   = [stats[g]['r'] for g in grps]
            bars = ax2.bar(grps, rs, color=[gc[g] for g in grps], alpha=0.85, width=0.5)
            ax2.axhline(0, color='#444', lw=1)
            ax2.axhspan(-1, -0.3, alpha=0.06, color='#2ecc71',
                        label='strong negative (prediction)')
            ax2.set_ylim(-1.05, 1.05)
            ax2.set_ylabel("Pearson r  (silence depth vs manifold dist)", color='#888', fontsize=9)
            sig_sym = '**' if perm_p<.01 else ('*' if perm_p<.05 else 'n.s.')
            ax2.set_title(f"Prediction: healthy r < 0\nΔr permutation p={perm_p:.3f}  ({sig_sym})",
                         color='#ddd', fontsize=10)
            ax2.tick_params(colors='#666', labelsize=9)
            ax2.legend(facecolor='#1a1a28', edgecolor='#333', labelcolor='white', fontsize=7.5)
            for g, bar in zip(grps, bars):
                s = stats[g]
                sg = '**' if s['p']<.01 else ('*' if s['p']<.05 else 'n.s.')
                yoff = 0.04 if bar.get_height() >= 0 else -0.08
                ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+yoff,
                         f"p={s['p']:.3f}\n{sg}",
                         ha='center', va='bottom', color='white', fontsize=8)
            for sp in ax2.spines.values(): sp.set_color('#2a2a3e')

            fig.tight_layout(pad=2.4)
            parts = [f"{g}: r={stats[g]['r']:+.3f} "
                     f"({'**' if stats[g]['p']<.01 else '*' if stats[g]['p']<.05 else 'n.s.'}) "
                     f"n={stats[g]['n']}" for g in grps]
            summary = (f"{len(points)} events  |  " + "  |  ".join(parts) +
                       f"  |  Δr perm p={perm_p:.3f}")

            def show():
                pb.stop(); pb.pack_forget()
                cv = FigureCanvasTkAgg(fig, master=frame)
                cv.draw()
                cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                stat_lbl.config(text=summary)
                self._set_status("Sub-second δ-coding: " + summary)
            self.root.after(0, show)

        except Exception as e:
            import traceback; traceback.print_exc()
            self.root.after(0, lambda: [pb.stop(), stat_lbl.config(text=f"Error: {e}")])

    threading.Thread(target=run, daemon=True).start()


App.run_subsecond_analysis = _run_subsecond_analysis


if __name__=='__main__':
    root=tk.Tk(); App(root); root.mainloop()
