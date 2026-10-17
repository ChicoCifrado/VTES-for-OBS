#!/usr/bin/env python3
"""Valida el embedder entrenado contra los warps reales de la camara.

Uso: python3 validate_embedder.py --onnx X --bin Y --meta Z [--frames DIR] [--target-id 100680]
"""
import argparse, json, os, glob, sys
import numpy as np
import cv2

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Recorte de arte NCC (identico a build_embedding_index.py / embedding_matcher.h)
NCC_ART_Y0, NCC_ART_Y1 = 0.08, 0.64
NCC_ART_X0, NCC_ART_X1 = 0.03, 0.97

def preprocess(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    x = img.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1)

def ncc_from_path(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    art = img[int(h*NCC_ART_Y0):int(h*NCC_ART_Y1), int(w*NCC_ART_X0):int(w*NCC_ART_X1)]
    art = cv2.resize(art, (32, 32), interpolation=cv2.INTER_AREA)
    v = art.astype(np.float32).flatten()
    v -= v.mean()
    nv = np.linalg.norm(v)
    return v / nv if nv > 1e-6 else v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', required=True)
    ap.add_argument('--bin', required=True)
    ap.add_argument('--meta', required=True)
    ap.add_argument('--frames', default='/mnt/c/Users/JackSuicide/AppData/Roaming/obs-studio/plugin_config/vtes-card-scanner/ocr_debug')
    ap.add_argument('--target-id', default='100680')
    ap.add_argument('--topk', type=int, default=5)
    args = ap.parse_args()

    import onnxruntime as ort
    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = ort.InferenceSession(args.onnx, so, providers=['CPUExecutionProvider'])
    in_name = sess.get_inputs()[0].name

    meta = json.load(open(args.meta, encoding='utf-8'))
    embs = np.fromfile(args.bin, dtype=np.float32).reshape(len(meta), -1)
    dim = embs.shape[1]
    fused = (dim == 4 * 1024 + 3 * 32 * 32)
    if not fused:
        # renormalizar por si acaso (modo embeddings puro)
        n = np.linalg.norm(embs, axis=1, keepdims=True); n[n < 1e-6] = 1.0
        embs = embs / n
    ids = [str(m.get('id')) for m in meta]

    frames = sorted(glob.glob(os.path.join(args.frames, 'vtes_ocr_card_*.png')),
                    key=lambda f: int(f.split('_')[-1].split('.')[0]))
    print(f'frames: {len(frames)}, index: {len(meta)} cards, dim={dim} fused={fused}')
    ranks = []
    for f in frames:
        x = preprocess(f)
        if x is None:
            continue
        out = sess.run(None, {in_name: x[None]})[0][0]
        q = out / (np.linalg.norm(out) + 1e-9)
        if fused:
            q = np.concatenate([np.tile(q, 4), ncc_from_path(f)]).astype(np.float32)
        sims = embs @ q
        order = np.argsort(-sims)
        pos = [i for i in order if ids[i] == args.target_id]
        rank = 1 + int(np.where(order == pos[0])[0][0]) if pos else -1
        ranks.append(rank)
        top = order[:args.topk]
        tag = 'OK ' if rank == 1 else '   '
        tops = ', '.join(f'{meta[t]["name"]}({sims[t]:.3f})' for t in top[:3])
        print(f'{tag}{os.path.basename(f):22s} rank={rank:4d} target_sim={sims[pos[0]]:.3f} | {tops}')
    if ranks:
        r = np.array(ranks)
        print(f'\nRESUMEN: top1={int((r==1).sum())}/{len(r)}  top3={int((r<=3).sum())}/{len(r)}  '
              f'mediana={int(np.median(r))}  media={r.mean():.1f}')

if __name__ == '__main__':
    main()
