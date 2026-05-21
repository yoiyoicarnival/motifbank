---
title: HallucinationInspectionMachine
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: AIの幻覚を数式で検出 — γ(r,d) Universal Law
---

# 🔬 HallucinationInspectionMachine

AIへの質問が「確実に答えられる領域」か「幻覚リスクがある領域」かを数式で診断します。

## 使い方

1. テキストボックスに質問を入力
2. 「診断する」ボタンを押す
3. 🟢 SAFE / 🟡 CREATIVE / 🔴 RISKY / 🟣 DANGER が表示される

## 理論

```
γ(d) = 1 - exp(-k · max(d - r_th, 0))    [k=0.405, r_th=0.283]
I(d) = γ(1-γ) = Var[Bernoulli(γ)]         [情報生成ポテンシャル]
d*   = r_th + ln2/k = 1.994               [最適アイデア生成点]
```

フラクタル幾何学からLLM知識空間まで同じ法則が成立する「知識境界の普遍理論」。

## 精度

- AUC = 0.857
- Precision = 1.000 (偽陽性ゼロ)
- 7システム (フラクタル〜LLM) で検証済み
