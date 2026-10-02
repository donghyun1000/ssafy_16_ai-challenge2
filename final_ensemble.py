"""
최종 제출 재현 스크립트 (CPU · 수 초)

각 모델이 test 6,714문항에 대해 저장한 a/b/c/d 확률(.npy)만으로
최종 선택한 두 제출물을 다시 만든다.

  eq_all       : 6개 모델 확률의 단순 평균 (같은 비중)      -> 공개 0.96574
  m3_f3_32b25  : 1024px 중심 가중 평균 + 32B 25%            -> 공개 0.96455

사용법
  python final_ensemble.py --inputs ./ensemble_inputs --data ./data --out ./submissions
  python final_ensemble.py --inputs ./ensemble_inputs --data-zip ./data.zip --out ./submissions

--inputs 폴더에는 아래 MODELS 의 파일 이름(.npy)이 있어야 한다.
(한글 파일 이름의 자모 분리 · '.npy' 누락 · '의 사본' 꼬리표는 자동으로 정리해서 찾는다)
"""
import argparse
import io
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# 키 : 파일 이름 (확장자 제외)  — 모델 설명은 README 의 '최종 파이프라인' 표 참고
MODELS = {
    "동현TTA":  "동현_TTA_8B_640px",         # Qwen3-VL-8B bf16 · 640px · 전량 2ep + TTA 6조합      (단독 0.9568)
    "수혁TTA4": "수혁_TTA_8B_640px",         # Qwen3-VL-8B 4bit · 640px · 증강 · 전량 2ep + TTA 4조합 (단독 0.95472)
    "수혁TTA9": "수혁_TTA9_8B_640px",        # 위 모델 + TTA 9조합                                    (m3 에만 사용)
    "B3":       "수혁_B3_3B_px1024",         # Qwen2.5-VL-3B · 1024px · 1,811개                       (단독 0.9288)
    "8B1024s":  "동현_8B_1024px",            # Qwen3-VL-8B 4bit · 1024px · 2,000개 · 1ep              (단독 0.95531)
    "1024full": "동현_TTA_8B_1024px_full",   # Qwen3-VL-8B 4bit · 1024px · 전량 2ep + TTA             (단독 0.95770)
    "32B":      "동현_32B_1024px",           # Qwen3-VL-32B 4bit · 1024px · 6,700개 · 1ep             (단독 0.95710)
}

RECIPES = {
    "eq_all":      {"동현TTA": 1, "수혁TTA4": 1, "B3": 1, "8B1024s": 1, "1024full": 1, "32B": 1},
    "m3_f3_32b25": {"1024full": .30, "동현TTA": .15, "수혁TTA9": .15, "B3": .075, "8B1024s": .075, "32B": .25},
}

LETTERS = np.array(list("abcd"))


def clean(name: str) -> str:
    n = unicodedata.normalize("NFC", name).replace("의 사본", "").strip()
    return n[:-4] if n.endswith(".npy") else n


def load_probs(inputs: Path, n_test: int) -> dict:
    files = {clean(p.name): p for p in inputs.iterdir() if p.is_file()}
    probs = {}
    for key, stem in MODELS.items():
        f = files.get(unicodedata.normalize("NFC", stem))
        if f is None:
            print(f"  (없음) {stem}")
            continue
        a = np.load(f, allow_pickle=False).astype(np.float64)
        assert a.shape == (n_test, 4), f"{stem}: 모양 {a.shape} != ({n_test}, 4)"
        probs[key] = a / a.sum(1, keepdims=True)
        print(f"  {key:<9} {f.name}")
    return probs


def load_ids(data: Path = None, data_zip: Path = None):
    if data is not None:
        return pd.read_csv(data / "test.csv"), pd.read_csv(data / "sample_submission.csv")
    with zipfile.ZipFile(data_zip) as z:
        pick = lambda fn: next(n for n in z.namelist() if n.endswith(fn) and "__MACOSX" not in n)
        return (pd.read_csv(io.BytesIO(z.read(pick("test.csv")))),
                pd.read_csv(io.BytesIO(z.read(pick("sample_submission.csv")))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, type=Path, help="모델별 test 확률 .npy 폴더")
    ap.add_argument("--data", type=Path, help="test.csv · sample_submission.csv 가 있는 폴더")
    ap.add_argument("--data-zip", type=Path, help="또는 data.zip")
    ap.add_argument("--out", default=Path("submissions"), type=Path)
    args = ap.parse_args()
    assert args.data or args.data_zip, "--data 또는 --data-zip 중 하나가 필요합니다"

    test_df, sample = load_ids(args.data, args.data_zip)
    id_col, ans_col = sample.columns[0], sample.columns[1]
    print(f"test {len(test_df)}문항 | 모델 확률 불러오기:")
    probs = load_probs(args.inputs, len(test_df))

    args.out.mkdir(parents=True, exist_ok=True)
    answers = {}
    for tag, w in RECIPES.items():
        miss = [k for k in w if k not in probs]
        if miss:
            print(f"[{tag}] 재료 부족으로 건너뜀: {miss}")
            continue
        ww = np.array(list(w.values()), dtype=float); ww /= ww.sum()
        avg = sum(x * probs[k] for x, k in zip(ww, w))
        ans = LETTERS[avg.argmax(1)]
        sub = sample.copy()
        sub[ans_col] = sub[id_col].map(dict(zip(test_df[id_col], ans)))
        assert sub[ans_col].isna().sum() == 0
        path = args.out / f"submission_{tag}.csv"
        sub.to_csv(path, index=False)
        answers[tag] = ans
        dist = pd.Series(ans).value_counts().sort_index().to_dict()
        print(f"[{tag}] 저장 {path} | 답 분포 {dist}")

    if len(answers) == 2:
        a, b = answers.values()
        print(f"두 제출물이 다른 답: {int(np.sum(a != b))}개")


if __name__ == "__main__":
    main()
