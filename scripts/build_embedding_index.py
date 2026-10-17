#!/usr/bin/env python3
"""
build_embedding_index.py — genera embeddings_1024d.bin + embeddings_1024d_meta.json
para el EmbeddingMatcher del plugin VTES Card Scanner.

Replica EXACTAMENTE el preprocesado de runtime (src/embedding_matcher.h):
  BGR -> RGB, resize 224x224 INTER_LINEAR, /255, normalizacion ImageNet por canal,
  forward -> embedding 1024d, normalizacion L2.

Modos:
  * normal: una fila de 1024 floats por entrada (variante limpia).
  * --variants clean,l1,l2,l3: una fila por variante (Nx1024). Cada variante
    degrada la referencia (blur + oscurecido + glow) para imitar la camara.
  * --fused: UNA fila por entrada de (4*1024 + 3072) floats:
        [0.25*emb_clean, 0.25*emb_l1, 0.25*emb_l2, 0.25*emb_l3, ncc(32x32x3)]
    El runtime construye la consulta [emb, emb, emb, emb, ncc_q] de modo que el
    producto punto = media de las 4 similitudes de embedding + similitud NCC del
    recorte de arte. Fusion validada 20/20 en frames reales de camara.

Uso:
  python3 build_embedding_index.py --onnx data/vtes_embedder_1024d.onnx \
      --vtes-json data/vtes.json --output data/ --fused \
      [--cache data/card-cache] [--cache-es data/card-cache-es] \
      [--validate IMG...]
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import cv2

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# nombre -> (sigma_blur, factor_oscuridad, sigma_glow) o None (limpio)
VARIANTS = {
    'clean': None,
    'l1': (1.5, 0.60, 4.0),
    'l2': (2.5, 0.45, 6.0),
    'l3': (4.0, 0.35, 8.0),
}

# Recorte de arte para el bloque NCC (identico en runtime, embedding_matcher.h)
NCC_ART_Y0, NCC_ART_Y1 = 0.08, 0.64
NCC_ART_X0, NCC_ART_X1 = 0.03, 0.97
NCC_EMB_SCALE = 0.25   # peso de cada bloque de embedding dentro de la fila fusionada


def safe(s):
    s = str(s).strip().replace(' ', '_').replace("'", '').replace('"', '')
    return ''.join(c for c in s if c.isalnum() or c in '_-.')


def degrade(img, sg, dark, gl):
    d = cv2.GaussianBlur(img, (0, 0), sg)
    d = (d.astype(np.float32) * dark).astype(np.uint8)
    g = cv2.GaussianBlur(d, (0, 0), gl)
    return cv2.addWeighted(d, 0.7, g, 0.6, 0)


def prep_from_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    x = img.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1)


def ncc_from_img(img):
    """Vector NCC (32x32x3, BGR intercalado, media 0, norma 1) del recorte de arte."""
    h, w = img.shape[:2]
    y0, y1 = int(h * NCC_ART_Y0), int(h * NCC_ART_Y1)
    x0, x1 = int(w * NCC_ART_X0), int(w * NCC_ART_X1)
    art = img[y0:y1, x0:x1]
    art = cv2.resize(art, (32, 32), interpolation=cv2.INTER_AREA)
    v = art.astype(np.float32).flatten()
    v -= v.mean()
    nv = np.linalg.norm(v)
    return v / nv if nv > 1e-6 else v


def preprocess(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return prep_from_img(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', required=True)
    ap.add_argument('--vtes-json', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--cache', default=None)
    ap.add_argument('--cache-es', default=None)
    ap.add_argument('--variants', default='clean',
                    help='lista separada por comas: clean,l1,l2,l3')
    ap.add_argument('--fused', action='store_true',
                    help='fila unica por entrada: 4 bloques emb (x0.25) + bloque NCC')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--validate', nargs='*', default=[])
    args = ap.parse_args()

    variant_names = [v.strip() for v in args.variants.split(',') if v.strip()]
    for v in variant_names:
        if v not in VARIANTS:
            print(f'[build] variante desconocida: {v}', flush=True); sys.exit(2)
    if args.fused and len(variant_names) != 4:
        print('[build] --fused requiere exactamente 4 variantes (clean,l1,l2,l3)', flush=True)
        sys.exit(2)
    print(f'[build] variantes: {variant_names}  fused={args.fused}', flush=True)

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(args.onnx, so, providers=['CPUExecutionProvider'])
    in_name = sess.get_inputs()[0].name

    cards = json.load(open(args.vtes_json, encoding='utf-8'))
    cache = Path(args.cache) if args.cache else Path(args.vtes_json).parent / 'card-cache'
    cache_es = Path(args.cache_es) if args.cache_es else Path(args.vtes_json).parent / 'card-cache-es'

    entries = []   # (id, name, image_path)
    missing = 0
    for c in cards:
        cid = str(c.get('id', ''))
        name = c.get('printed_name') or c.get('_name') or c.get('name') or ''
        if not cid or not name:
            continue
        f = cache / (safe(c.get('_name') or name) + '.jpg')
        if f.exists() and f.stat().st_size > 1000:
            entries.append((cid, name, f))
        else:
            missing += 1
        es = ((c.get('_i18n') or {}).get('es') or {})
        if es.get('url'):
            fe = cache_es / (safe(c.get('_name') or name) + '.jpg')
            if fe.exists() and fe.stat().st_size > 1000:
                entries.append((cid, name, fe))
    print(f'[build] cards={len(cards)} entries={len(entries)} missing_default={missing}', flush=True)

    rows_list = []
    meta = []
    B = args.batch
    done = 0
    for i in range(0, len(entries), B):
        batch = entries[i:i+B]
        xs = []
        nccs = []
        batch_meta = []
        for cid, name, p in batch:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            for vname in variant_names:
                params = VARIANTS[vname]
                im2 = img if params is None else degrade(img, *params)
                xs.append(prep_from_img(im2))
            if args.fused:
                nccs.append(ncc_from_img(img))
            batch_meta.append((cid, name))
        if not xs:
            continue
        blob = np.stack(xs).astype(np.float32)
        out = sess.run(None, {in_name: blob})[0]
        n = np.linalg.norm(out, axis=1, keepdims=True)
        n[n < 1e-6] = 1.0
        out = out / n

        if args.fused:
            nv = len(variant_names)
            for k, (cid, name) in enumerate(batch_meta):
                emb_blocks = out[k*nv:(k+1)*nv] * NCC_EMB_SCALE      # 4 x 1024
                row = np.concatenate([emb_blocks.reshape(-1), nccs[k]])  # 4096 + 3072
                rows_list.append(row.astype(np.float32))
                meta.append({'id': cid, 'name': name})
        else:
            for k, (cid, name) in enumerate(batch_meta):
                for v in range(len(variant_names)):
                    rows_list.append(out[k*len(variant_names)+v].astype(np.float32))
                    meta.append({'id': cid, 'name': name})
        done += len(batch)
        if (i // B) % 20 == 0:
            print(f'[build] {done}/{len(entries)} imgs, {len(meta)} rows', flush=True)

    rows = np.stack(rows_list) if rows_list else np.zeros((0, 1024), np.float32)
    out_bin = Path(args.output) / 'embeddings_1024d.bin'
    out_meta = Path(args.output) / 'embeddings_1024d_meta.json'
    rows.tofile(out_bin)
    json.dump(meta, open(out_meta, 'w', encoding='utf-8'), ensure_ascii=False)
    print(f'[build] wrote {out_bin} ({rows.nbytes/1024/1024:.1f} MB, {rows.shape[0]}x{rows.shape[1]}) '
          f'and {out_meta} ({len(meta)} rows)', flush=True)

    # Validacion opcional contra imagenes reales de camara
    dim = rows.shape[1]
    for vp in args.validate:
        img = cv2.imread(str(vp), cv2.IMREAD_COLOR)
        if img is None:
            print(f'[val] {vp}: no se pudo leer'); continue
        o = sess.run(None, {in_name: prep_from_img(img)[None]})[0][0]
        o = o / (np.linalg.norm(o) + 1e-9)
        if args.fused:
            q = np.concatenate([np.tile(o, 4), ncc_from_img(img)]).astype(np.float32)
        else:
            q = o.astype(np.float32)
        sims = rows @ q
        order = np.argsort(-sims)
        seen = set(); shown = 0
        print(f'[val] {vp}:')
        for t in order:
            key = meta[t]['id']
            if key in seen:
                continue
            seen.add(key)
            print(f'    {sims[t]:.3f}  id={key}  {meta[t]["name"]}')
            shown += 1
            if shown >= 5:
                break


if __name__ == '__main__':
    main()
