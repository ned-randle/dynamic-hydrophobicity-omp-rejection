# -*- coding: utf-8 -*-
"""计算并展示模型的 SHAP 解释结果。"""

# =========================
# 1) 路径与基础配置
# =========================
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data.csv"
MODEL_PATH = PROJECT_DIR / "model.joblib"
OUTPUT_DIR = PROJECT_DIR / "shap_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "<特征列_1>",
    "<特征列_2>",
    ...
]


# =========================
# 2) 读取分析数据
# =========================
def load_data(path: Path) -> pd.DataFrame:
    """读取用于模型解释的数据。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件: {path}")

    data = pd.read_csv(path)
    return data.copy()


# =========================
# 3) 加载模型并处理输入特征
# =========================
def load_model_and_transform(
    data: pd.DataFrame,
    model_path: Path,
) -> tuple[object, pd.DataFrame]:
    """加载训练好的模型，并复用训练阶段的预处理步骤。"""
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型文件: {model_path}")

    fitted_pipeline = joblib.load(model_path)
    X = data[FEATURE_COLS]

    if hasattr(fitted_pipeline, "named_steps"):
        steps = fitted_pipeline.named_steps
        preprocessor = steps.get("imputer")
        if preprocessor is None:
            preprocessor = steps.get("preprocessor")

        model = steps.get("xgb")
        if model is None:
            model = steps.get("model")

        if model is None:
            raise KeyError("Pipeline 中未找到模型步骤，请检查步骤名称。")

        if preprocessor is not None:
            X = pd.DataFrame(
                preprocessor.transform(X),
                columns=FEATURE_COLS,
                index=data.index,
            )
    else:
        model = fitted_pipeline

    return model, X


# =========================
# 4) 计算 SHAP 值
# =========================
def calculate_shap_values(model: object, X: pd.DataFrame) -> shap.Explanation:
    """创建解释器并计算每个样本的特征贡献。"""
    explainer = shap.TreeExplainer(model)
    return explainer(X)


# =========================
# 5) 绘制全局特征重要性
# =========================
def save_importance_plot(shap_values: shap.Explanation) -> None:
    """保存基于平均绝对 SHAP 值的全局重要性图。"""
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_feature_importance.png", dpi=300)
    plt.close()


# =========================
# 6) 绘制 SHAP 分布图
# =========================
def save_summary_plot(shap_values: shap.Explanation) -> None:
    """保存特征贡献方向和分布的汇总图。"""
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_summary.png", dpi=300)
    plt.close()


# =========================
# 7) 绘制单特征依赖关系
# =========================
def save_dependence_plot(
    shap_values: shap.Explanation,
    X: pd.DataFrame,
    feature: str | None,
) -> None:
    """展示指定特征取值与其 SHAP 贡献之间的关系。"""
    if feature is None:
        return
    if feature not in X.columns:
        raise KeyError(f"依赖图特征不在输入数据中: {feature}")

    shap.dependence_plot(
        feature,
        shap_values.values,
        X,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"shap_dependence_{feature}.png", dpi=300)
    plt.close()


# =========================
# 8) 运行 SHAP 分析流程
# =========================
def main() -> None:
    """加载数据和模型，计算 SHAP 值并保存主要图件。"""
    data = load_data(DATA_PATH)
    model, X = load_model_and_transform(data, MODEL_PATH)
    shap_values = calculate_shap_values(model, X)

    save_importance_plot(shap_values)
    save_summary_plot(shap_values)
    save_dependence_plot(shap_values, X, DEPENDENCE_FEATURE)

    print(f"SHAP 结果已保存至: {OUTPUT_DIR}")

