======================================================================
## MotifBank × MFI Silicalite-1: スケーリング解析結果
## (2026-05-17〜18, si_oh4, PBE/def2-SVP, R_cut=5.5Å, MIC修正済)
======================================================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
◆ 対象系
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  MFI silicalite-1 (pure SiO2 zeolite)
  ソース    : IZA Structure Database  http://www.iza-structure.org/
  CIF       : MFI_iza.cif (DLS76最適化, pure SiO2)
  単位セル  : 96 Si + 192 O = 288 原子, Pnma, a=20.07 b=19.92 c=13.42 Å
  断片化    : Si(OH)4 (mol_type=si_oh4, Si + 最近傍 4O + 4H キャップ)
  T-sites   : 12 種の結晶学的非等価 Si 位置

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
◆ Phase 分類 (MIC修正後)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Phase        : 0  (N_bank 飽和)
  N_bank_sat   : 282 型 (1x1x1 から即飽和)
    内訳: um=1 (全モノマー同一テトラヘドロン)
          up=93 (unique ペア)
          ut=188 (unique トリマー)
  戦略         : DEPLOY

  解釈: 全 Si(OH)4 モノマーは近似的に同一の正四面体。
        局所多様性はペア・トリマー間の幾何差異に起因。
        N を無限大にしても bank は 282 型以上増えない。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
◆ QC コール数 比較 (実測, MIC修正後)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  スーパーセル   N_SiO4   QC_naive              QC_bank  speedup
  ──────────────────────────────────────────────────────────────
  1x1x1              96    1,450  ( 96+490+864)       282     5x
  2x2x1             384    6,682  (384+2196+4102)     282    24x
  2x2x2             768   14,690  (768+4748+9174)     282    52x
  4x2x2           1,536   29,690                      282   105x

  ※ QC_bank = 282 で固定 (1x1x1 から即飽和)
  ※ QC_naive = モノマー + ペア + トリマー QC 計算回数

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
◆ スケーリング外挿 (bank=282 固定, naive は N^1.1 スケール)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  N_SiO4     QC_naive (推定)   QC_bank   speedup
  ──────────────────────────────────────────────
       96             1,450       282        5x   (実測)
      384             6,682       282       24x   (実測)
      768            14,690       282       52x   (実測)
    1,536            29,690       282      105x   (実測)
    3,072           ~65,000       282      ~230x
    6,144          ~140,000       282      ~497x
   10,000          ~240,000       282      ~851x
  100,000        ~3,000,000       282   ~10,638x

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
◆ PBE/def2-SVP wall-clock speedup (実測 T_QC=12.5s/call)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  スーパーセル   N_SiO4   T_naive      T_bank   speedup
  ──────────────────────────────────────────────────────
  1x1x1              96      5.1 h      1.0 h       5x
  2x2x1             384     23.3 h      1.0 h      24x
  2x2x2             768     51.2 h      1.0 h      52x
  4x2x2           1,536    103.4 h      1.0 h     105x
  (外挿) 10,000         836 h      1.0 h     851x

  ※ T_QC = 12.5s/call (PBE/def2-SVP, Si(OH)4, OMP_NUM_THREADS=1)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
◆ 物理的解釈
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  MFI は 12 種の結晶学的に非等価な T サイト (Si 位置) を持つ。
  MIC 修正後、全モノマーが同一テトラヘドロン (um=1) と判明。
  局所多様性はペア・トリマー接続パターンで決まる (up=93, ut=188)。

  MotifBank の speedup = N / N_bank_sat = N / 282

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
◆ 重要バグ修正: PBC (MIC) 対応 (2026-05-18)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  問題: from_cif (sio4/si_oh4) が Cartesian 距離を使用。
        周期境界をまたぐ Si-O 結合を誤認 → 3.37 Å の「偽結合」
        → 歪んだジオメトリ・過剰な N_bank。

  修正: minimum image convention (MIC) を実装。
        np.linalg.solve(cell.T, diff) → frac → round → Cartesian 変換。

  修正前 → 修正後:
    MFI N_bank:         644 → 282
    speedup (N=768):    22x → 52x
    cristobalite N_bank: ~400 (推定) → 18 (実測・飽和確認)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
◆ H-capped Si(OH)4 断片化
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  問題: SiO4 は [SiO4]^4- (charge=-4), charge=0 では SCF 未収束
  解決: dangling bond を H でキャップ → Si(OH)4 (中性, charge=0)

  実装: mol_type="si_oh4"
        Si + 4O (最近傍, MIC距離) + 4H (O-Si 延長上 0.96Å)

  検証 (test_sioh4.py, 7/7 PASS):
    ✅ F1a-e: 9原子/断片, Si-O=1.608~1.611 Å, O-H=0.960 Å
    ✅ F2: PySCF HF/STO-3G charge=0 収束  E=-583.221734 Ha
    ✅ F3: Phase 0 維持 (si_oh4), gamma=0.063, DEPLOY
    ✅ F4: N_bank 飽和値 = 282, S_local=5.64 nats
    ✅ F5: sio4 vs si_oh4 → bank(sio4)=281, bank(si_oh4)=282 (ほぼ同値)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
◆ S_local: ローカル幾何エントロピー (新概念)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  定義: S_local(material) = log( N_bank_sat )
        N_bank_sat = bank が飽和したときのエントリ数

  物理的意味:
    - S_local が有限 = 局所幾何の種類が有限 = 周期結晶の定義そのもの
    - S_local が小さい材料 = MotifBank の効果が大きい
    - S_local = ∞ (= log N) のとき Phase 3 (ランダム系)

  材料による S_local 実測値 (MIC修正後, 全て実測):
    材料                     N_bank_sat   S_local      Phase
    ──────────────────────────────────────────────────────
    ice Ih (3x3 CIF)              16      2.77 nats    0  ✅ (H2O, 1x→8x で飽和)
    alpha-cristobalite            18      2.89 nats    0  ✅ (SiO2, 1T-site, sat 4x)
    LTA zeolite (si_oh4)          66      4.19 nats    0  ✅ (1T-site cubic, sat 4x)
    MFI silicalite-1             282      5.64 nats    0  ✅ (12T-sites, sat 1x1x1)
    defect MFI (M=12 vocab)    ~1800     ~7.5 nats    1  ~ (γ=0.18, sub-linear)
    amorphous Si(OH)4             ∞       → ∞          3  (γ=1.56, linear)

  ※ ice ≈ cristobalite (どちらも1T-site 4配位, 小単位胞)
  ※ LTA < MFI は T-site 数 (1 vs 12) と対称性の差を反映
  ※ S_local の物理: T-site 数 × 局所対称性 が決定

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
◆ N_bank(N) スケーリング図: 3フェーズ (2026-05-18)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  motifbank_scaling_figure.png  (log-log, crystal/defect/amorphous)

  結果:
    crystal (MFI)    γ = 0.086  Phase 0 (飽和)
    defect (M=12)    γ = 0.177  Phase 1 (sub-linear, 誕生日問題的)
    amorphous        γ = 1.555  Phase 3 (linear)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
◆ 精度・正確性検証結果
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  HF/STO-3G 検証 (test_mfi_accuracy.py, 7/7 PASS):
    A1. naive MBE == bank MBE       ΔE = 0.00e+00 Ha
    A2. memory_saving=True == False  ΔE = 0.00e+00 Ha
    A3. geom_key 順序不変性          pair 10件すべて一致
    A4. bank ROI (2回目)            100.0%
    T1. 壁時間 speedup               1.8x  (5ms/call delay, N=96)
    T2. QC call 削減比               346x  (naive=346, bank=0 on 2nd run)
    N1. SiO4 charge=-4              E = -578.485385 Ha (HF/STO-3G)

  PBE/def2-SVP 検証 (test_pbe_wallclock.py, 3/3 PASS):
    P1. Si(OH)4 収束                E = -592.129475 Ha, T=12.5s/call
    P2. wall-clock speedup > 10x   52x at N=768 (naive 51h → bank 1h)
    P3. naive == bank (PBE)        ΔE = 0.00e+00 Ha ✅ (toy 卒業)
        E_naive(N=5) = -2960.647373 Ha
        bank 2nd run: 0.001s vs naive 11.7s → 11,661x

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
◆ 現状の限界と次のステップ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  実証済み:
    ✅ Phase 0 確認 (MFI: N_bank=282, 即飽和)
    ✅ QC コール削減 52x 実測 (N=768)
    ✅ naive == bank  ΔE=0 (HF/STO-3G AND PBE/def2-SVP)
    ✅ PBE/def2-SVP 収束 (T_QC=12.5s/call, toy 卒業)
    ✅ S_local 実測 4材料 (ice/cristobalite/LTA/MFI)
    ✅ 3フェーズ スケーリング図 (crystal/defect/amorphous)
    ✅ memory_saving モード正確性
    ✅ PBC (MIC) バグ修正

  残っていること:
    ❌ 物理精度: MBE 打ち切り誤差 vs 周期 DFT (VASP/CP2K)
                 目標: < 1 kcal/mol / SiO4
    ❌ 第三者ベンチマーク: 望月先生(立教大)のFMOデータ待ち
                 → compare_fmo_benchmark.py で即対応可能
    △ 一部 Si(OH)4 断片が PBE/def2-SVP で未収束
                 (SCF困難なジオメトリ, Newton法でも失敗)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
◆ 再現コマンド
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # CIF 入手
  curl -o examples/MFI_iza.cif 'http://www.iza-structure.org/IZA-SC/cif/MFI.cif'
  curl -o examples/LTA_iza.cif 'http://www.iza-structure.org/IZA-SC/cif/LTA.cif'

  # Phase 分類
  OMP_NUM_THREADS=1 python3 motifbank_cli.py classify examples/MFI_iza.cif

  # N_bank(N) スケーリング図 (3フェーズ)
  OMP_NUM_THREADS=1 python3 motifbank_scaling_figure.py --plot

  # S_local 比較図 (4材料バーチャート)
  python3 s_local_figure.py --plot

  # PBE/def2-SVP wall-clock ベンチマーク (P1/P2 only)
  OMP_NUM_THREADS=1 python3 test_pbe_wallclock.py

  # PBE/def2-SVP ΔE=0 実証 (P3, ~10 min)
  OMP_NUM_THREADS=1 python3 test_pbe_wallclock.py --n5

  # FMO ベンチマークとの比較 (データ受領後)
  OMP_NUM_THREADS=1 python3 compare_fmo_benchmark.py --ref ref.csv --system MFI

  # 全単体テスト
  OMP_NUM_THREADS=1 python3 test_sioh4.py
  OMP_NUM_THREADS=1 python3 test_mfi_accuracy.py

======================================================================
