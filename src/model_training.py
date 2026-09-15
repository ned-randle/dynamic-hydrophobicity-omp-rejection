# -*- coding: utf-8 -*-
"""
模型训练与评估流程。
"""

# =========================
# 1) 路径与基础配置
# =========================
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data.csv"
OUTPUT_DIR = PROJECT_DIR / "File" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 保留“多随机种子重复训练”的结构；数量可按实验要求调整。
RANDOM_SEEDS = range(10)
TRAIN_SIZE = 0.8
CV_SPLITS = 5


# =========================
# 2) 特征与目标定义
# =========================
BASE_INPUT_COLS = [
    "<特征列_1>",
    "<特征列_2>",
    ...
]
OBJ_COL = "<目标列>"


# =========================
# 3) 读取与筛选数据
# =========================
def load_data(path: Path) -> pd.DataFrame:
    """读取建模数据，并按需要保留训练样本。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件: {path}")

    data = pd.read_csv(path)
    return data.copy()


# =========================
# 4) 指标辅助函数
# =========================
def calc_adjusted_r2(r2: float, n_samples: int, n_features: int) -> float:
    """根据样本量和特征数计算调整后的 R²。"""
    if n_samples <= n_features + 1:
        return r2
    return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)


# =========================
# 5) 预处理与模型流水线
# =========================
def build_pipeline(params: dict, random_state: int) -> Pipeline:
    model_kwargs = {
        **params,
        "random_state": random_state,
        "n_jobs": 1,
    }
    
    model = xgb.XGBRegressor(**model_kwargs)
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("model", model),
        ]
    )


# =========================
# 6) 超参数优化
# =========================
def optimize_hyperparameters(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int,
) -> tuple[dict, float]:
    placeholder_params = {
        "n_estimators",
        "learning_rate",
        ...
    }

    pipeline = build_pipeline(placeholder_params, random_state=random_state)
    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=random_state)
    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="r2",
        n_jobs=-1,
    )
    return placeholder_params, float(np.mean(scores))


# =========================
# 7) 单个随机种子的训练与评估
# =========================
def run_one_seed(data: pd.DataFrame, seed: int) -> dict:
    """完成一次数据切分、调参、训练、预测和指标计算。"""
    train_data, test_data = train_test_split(
        data,
        train_size=TRAIN_SIZE,
        random_state=seed,
    )

    X_train = train_data[BASE_INPUT_COLS]
    y_train = train_data[OBJ_COL].to_numpy().ravel()
    X_test = test_data[BASE_INPUT_COLS]
    y_test = test_data[OBJ_COL].to_numpy().ravel()

    # 先通过交叉验证选择参数，再使用训练集拟合最终模型。
    best_params, best_cv_r2 = optimize_hyperparameters(
        X_train,
        y_train,
        random_state=seed,
    )
    final_pipeline = build_pipeline(best_params, random_state=seed)
    final_pipeline.fit(X_train, y_train)

    y_pred = final_pipeline.predict(X_test)

    n_features = len(BASE_INPUT_COLS)
    test_r2 = r2_score(y_test, y_pred)
    return {
        "seed": seed,
        "R2_cv_adj": calc_adjusted_r2(
            best_cv_r2,
            n_samples=max(1, len(y_train) // CV_SPLITS),
            n_features=n_features,
        ),
        "R2_test_adj": calc_adjusted_r2(
            test_r2,
            n_samples=len(y_test),
            n_features=n_features,
        ),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "best_params": best_params,
    }


# =========================
# 8) 随机种子循环与结果汇总
# =========================
def main() -> None:
    """运行模型训练与评估流程，并保存指标及参数摘要。"""
    started_at = time.time()
    data = load_data(DATA_PATH)

    results = []
    for seed in RANDOM_SEEDS:
        print(f"正在训练 Seed {seed} ...")
        results.append(run_one_seed(data, seed))

    # 指标表不包含嵌套参数，便于后续统计分析。
    metrics_df = pd.DataFrame(
        [{key: value for key, value in item.items() if key != "best_params"} for item in results]
    )
    metrics_df.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)

    # 按交叉验证指标选出代表性结果，并单独保存参数摘要。
    best_result = max(results, key=lambda item: item["R2_cv_adj"])
    with (OUTPUT_DIR / "selected_params.json").open("w", encoding="utf-8") as file:
        json.dump(best_result["best_params"], file, ensure_ascii=False, indent=2)

    print(
        "平均 CV 调整 R²: "
        f"{metrics_df['R2_cv_adj'].mean():.3f} ± {metrics_df['R2_cv_adj'].std():.3f}"
    )
    print(f"总耗时: {(time.time() - started_at) / 60:.2f} 分钟")


