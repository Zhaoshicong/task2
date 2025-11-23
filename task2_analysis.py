import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import r2_score


def ensure_output_dir(output_dir):
    """
    确保输出目录存在，如果不存在则创建
    
    Args:
        output_dir (str): 输出目录路径
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


def read_series(csv_path, date_col="Date", value_col="Value"):
    """
    从CSV文件读取时间序列数据
    
    Args:
        csv_path (str): CSV文件路径
        date_col (str): 日期列名，默认为"Date"
        value_col (str): 数值列名，默认为"Value"
    
    Returns:
        pd.Series: 以日期为索引的时间序列
    
    Raises:
        ValueError: 如果CSV文件缺少必需的列
    """
    df = pd.read_csv(csv_path)
    if date_col not in df.columns or value_col not in df.columns:
        raise ValueError("CSV 必须包含列: Date 和 Value")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    df = df.set_index(date_col)
    series = df[value_col].astype(float)
    # 尝试推断频率，若失败则默认按月对齐
    try:
        inferred = pd.infer_freq(series.index)
        if inferred is None:
            series.index = series.index.to_period("M").to_timestamp()
    except Exception:
        series.index = series.index.to_period("M").to_timestamp()
    return series


def plot_rolling_stats(series, window, output_path, title_suffix=""):
    """
    绘制原始序列及其滑动统计量（均值、标准差）的图形
    
    Args:
        series (pd.Series): 时间序列数据
        window (int): 滑动窗口大小
        output_path (str): 输出图片保存路径
        title_suffix (str): 标题后缀，默认为空字符串
    """
    rolling_mean = series.rolling(window=window, min_periods=12).mean()
    rolling_std = series.rolling(window=window, min_periods=12).std()
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=series.index, y=series.values, label="Series")
    sns.lineplot(x=rolling_mean.index, y=rolling_mean.values, label=f"Rolling Mean ({window})")
    sns.lineplot(x=rolling_std.index, y=rolling_std.values, label=f"Rolling Std ({window})")
    plt.title(f"原始序列与滑动统计{title_suffix}")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_adf(series):
    """
    执行ADF（Augmented Dickey-Fuller）平稳性检验
    
    Args:
        series (pd.Series): 待检验的时间序列
    
    Returns:
        tuple: (检验统计量, p值, 使用的滞后阶数, 观测数量, 临界值字典)
    """
    result = adfuller(series.dropna(), autolag="AIC")
    test_stat, pvalue, usedlag, nobs, crit_vals = (
        result[0], result[1], result[2], result[3], result[4]
    )
    return test_stat, pvalue, usedlag, nobs, crit_vals


def save_adf_report(series, output_path, title):
    """
    对时间序列执行ADF检验并将结果保存到文件
    
    Args:
        series (pd.Series): 待检验的时间序列
        output_path (str): 输出报告文件路径
        title (str): 报告标题（用于标识检验的对象）
    """
    test_stat, pvalue, usedlag, nobs, crit_vals = run_adf(series)
    crit_str = ", ".join([f"{k}: {round(v, 6)}" for k, v in crit_vals.items()])
    lines = [
        f"[{title}] ADF Test",
        f"Test Statistic: {test_stat:.6f}",
        f"p-value: {pvalue:.6f}",
        f"Used Lags: {usedlag}",
        f"Number of Observations: {nobs}",
        f"Critical Values: {" + crit_str + "}",
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def decompose_and_plot(series, model, period, output_prefix):
    """
    对时间序列进行季节分解（加性或乘性）并保存分解图和残差ADF检验报告
    
    Args:
        series (pd.Series): 时间序列数据
        model (str): 分解模型类型，"additive"（加性）或"multiplicative"（乘性）
        period (int): 季节周期（例如月度数据为12）
        output_prefix (str): 输出文件前缀路径
    """
    result = seasonal_decompose(series, model=model, period=period, extrapolate_trend="freq")
    fig = result.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle(f"{model.capitalize()} 分解 (period={period})", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_{model}_decompose.png", dpi=150)
    plt.close(fig)
    # 对残差做 ADF 并保存
    save_adf_report(result.resid.dropna(), f"{output_prefix}_{model}_resid_adf.txt", f"{model.capitalize()} Residuals")


def auto_determine_d(series, max_d=2, significance=0.05):
    """
    自动确定使序列平稳所需的最小差分阶数d
    
    通过ADF检验，从d=0开始逐步增加差分阶数，直到序列在指定显著性水平下平稳
    
    Args:
        series (pd.Series): 原始时间序列
        max_d (int): 最大允许的差分阶数，默认为2
        significance (float): 显著性水平，默认为0.05
    
    Returns:
        int: 使序列平稳的最小差分阶数d（如果达到max_d仍未平稳，则返回max_d）
    """
    for d in range(0, max_d + 1):
        if d == 0:
            test_series = series
        else:
            test_series = series.diff(d).dropna()
        _, pvalue, _, _, _ = run_adf(test_series)
        if pvalue < significance:
            return d
    return max_d


def plot_acf_pacf(series, lags, output_prefix):
    """
    绘制时间序列的自相关函数（ACF）和偏自相关函数（PACF）图
    
    这些图用于帮助选择ARIMA模型的p和q参数
    
    Args:
        series (pd.Series): 时间序列数据（通常是差分后的序列）
        lags (int): 要绘制的最大滞后阶数
        output_prefix (str): 输出图片文件的前缀路径
    """
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(series.dropna(), lags=lags, ax=ax[0])
    ax[0].set_title("ACF")
    plot_pacf(series.dropna(), lags=lags, ax=ax[1], method="ywm")
    ax[1].set_title("PACF")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_acf_pacf.png", dpi=150)
    plt.close()


def grid_arima(train, d, p_values, q_values):
    """
    对给定的p和q参数范围进行ARIMA模型网格搜索，返回按AIC排序的结果
    
    Args:
        train (pd.Series): 训练集时间序列
        d (int): 差分阶数（已确定）
        p_values (list): AR阶数p的候选值列表
        q_values (list): MA阶数q的候选值列表
    
    Returns:
        pd.DataFrame: 包含所有尝试的(p,d,q)组合及其AIC值的DataFrame，按AIC升序排序
    """
    rows = []
    for p in p_values:
        for q in q_values:
            order = (p, d, q)
            try:
                model = ARIMA(train, order=order)
                fitted = model.fit()
                aic = float(fitted.aic)
                rows.append({"p": p, "d": d, "q": q, "AIC": aic})
                print(f"已拟合 ARIMA{order}，AIC={aic:.2f}")
            except Exception as e:
                print(f"ARIMA{order} 拟合失败：{e}")
                rows.append({"p": p, "d": d, "q": q, "AIC": np.inf})
                continue
    metrics_df = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    return metrics_df


def main():
    """
    主函数：执行完整的时间序列分析流程
    
    流程包括：
    1. 读取训练数据
    2. 可视化原始序列和滑动统计
    3. ADF平稳性检验
    4. 季节分解（加性和乘性）
    5. 自动确定差分阶数d
    6. 绘制ACF/PACF图
    7. ARIMA模型网格搜索和选择（基于AIC）
    8. 在测试集上预测并评估（计算R²）
    9. 保存所有结果和可视化图表
    """
    parser = argparse.ArgumentParser(description="Task2: 时间序列分析与ARIMA预测（简化版）")
    parser.add_argument("--train", default="training.csv", help="训练集 CSV 路径，需含列 Date, Value")
    parser.add_argument("--test", default="testing.csv", help="测试集 CSV 路径，需含列 Date, Value")
    parser.add_argument("--output", default="outputs", help="输出目录")
    parser.add_argument("--period", type=int, default=12, help="季节分解周期（如月度数据取 12）")
    parser.add_argument("--rolling", type=int, default=12, help="滑动窗口大小")
    parser.add_argument("--max_p", type=int, default=2, help="ARIMA 最大 p（建议不大于 2）")
    parser.add_argument("--max_q", type=int, default=2, help="ARIMA 最大 q（建议不大于 2）")
    parser.add_argument("--max_d", type=int, default=2, help="自动选择 d 的最大阶数")
    args = parser.parse_args()

    ensure_output_dir(args.output)

    # 读取数据
    print("[1/7] 正在读取训练数据...")
    train_series = read_series(args.train)
    print(f"训练样本数: {len(train_series)}")

    # 可视化：原始序列与滑动统计
    print("[2/7]  生成原序列与滑动统计图...")
    plot_rolling_stats(train_series, args.rolling, os.path.join(args.output, "rolling_stats.png"))
    print("[3/7] 对原序列执行 ADF 检验并保存报告...")
    save_adf_report(train_series, os.path.join(args.output, "adf_original.txt"), "Original Series")

    # 分解：加性与乘性
    print("[4/7] 进行加性分解与残差检验...")
    decompose_and_plot(train_series, model="additive", period=args.period, output_prefix=os.path.join(args.output, "add"))
    print("[5/7] 进行乘性分解与残差检验...")
    decompose_and_plot(train_series, model="multiplicative", period=args.period, output_prefix=os.path.join(args.output, "mul"))

    # 自动确定差分阶数 d，并绘制 ACF/PACF（差分后的序列）
    print("[6/7] 自动确定差分阶数 d...")
    d = auto_determine_d(train_series, max_d=args.max_d)
    print(f"选择的差分阶数 d = {d}")
    diff_series = train_series.diff(d).dropna() if d > 0 else train_series
    print("绘制差分后序列的 ACF/PACF 图...")
    plot_acf_pacf(diff_series, lags=min(40, len(diff_series) // 2), output_prefix=os.path.join(args.output, f"diff_d{d}"))

    # 仅基于训练集进行 ARIMA 网格搜索与选择
    print("[7/7] 仅用训练集进行 ARIMA 网格搜索（按 AIC 选最优）...")
    p_values = list(range(0, args.max_p + 1))
    q_values = list(range(0, args.max_q + 1))
    metrics_df = grid_arima(train=train_series, d=d, p_values=p_values, q_values=q_values)
    metrics_csv = os.path.join(args.output, "arima_candidates_aic.csv")

    metrics_df.to_csv(metrics_csv, index=False)

    # 选择 AIC 最小的模型
    if metrics_df.empty or np.isinf(metrics_df.iloc[0]["AIC"]):
        print("未能成功拟合任何 ARIMA 模型。请检查数据或参数范围。")
        return

    best_row = metrics_df.iloc[0]
    best_order = (int(best_row["p"]), int(best_row["d"]), int(best_row["q"]))
    print(f"最佳模型为 ARIMA{best_order}，AIC={best_row['AIC']:.2f}")

    # 现在读取测试集，并用最佳模型做预测与评估
    try:
        test_series = read_series(args.test)
        horizon = len(test_series)
        fitted_best = ARIMA(train_series, order=best_order).fit()
        forecast_best = fitted_best.forecast(steps=horizon)
        y_pred = forecast_best.values
        pred_index = forecast_best.index
    except Exception as e:
        print(f"预测阶段出错：{e}")
        return

    # 计算 R²（测试集仅用于评估，不参与模型选择）
    y_true = test_series.reindex(pred_index).values
    r2 = r2_score(y_true, y_pred)
    print(f"测试集 R² = {r2:.4f}")

    # 保存预测对比图
    print("[8/8] 保存预测对比图与结果文件...")
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=train_series.index, y=train_series.values, label="Train")
    sns.lineplot(x=test_series.index, y=test_series.values, label="Test True")
    sns.lineplot(x=pred_index, y=y_pred, label=f"Forecast ARIMA{best_order}")
    plt.title(f"预测对比 | 最优ARIMA{best_order} | R²={r2:.4f} | AIC={best_row['AIC']:.2f}")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, "forecast_comparison.png"), dpi=150)
    plt.close()

    # 导出预测与指标
    pred_df = pd.DataFrame({
        "Date": pd.Index(pred_index, name="Date"),
        "y_true": y_true,
        "y_pred": y_pred,
    })
    pred_df.to_csv(os.path.join(args.output, "predictions.csv"), index=False)

    with open(os.path.join(args.output, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Best ARIMA order: {best_order}\n")
        f.write(f"Best AIC: {best_row['AIC']:.6f}\n")
        f.write(f"Test R2: {r2:.6f}\n")
        f.write(f"Chosen d by ADF: {d}\n")

    print("完成。输出已保存到:", os.path.abspath(args.output))


if __name__ == "__main__":
    main()


