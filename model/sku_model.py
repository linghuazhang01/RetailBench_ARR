import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fit_logit_demand_model(path, plot=False):
    """
    从单个 CSV 文件拟合 logit 需求模型参数：
        logit(share) = alpha + beta * price
    
    参数：
        path : str
            CSV 文件路径
        plot : bool, default False
            是否画模拟图（随机价格路径和真实散点等）
    
    返回：
        alpha, beta, sigma : float
            拟合得到的模型参数
    """
    # ===== 1. 读 & 清洗 =====
    df = pd.read_csv(path)
    df = df[(df["OK"] == 1) & (df["PRICE"] > 0) & (df["CUSTOMCOUNT"] > 0)].copy()
    df = df.sort_values("WEEK").reset_index(drop=True)

    if df.empty:
        raise ValueError("清洗后数据为空，请检查该 CSV 是否有有效的 OK/PRICE/CUSTOMCOUNT 记录。")

    # ===== 2. 拟合 alpha, beta, sigma （logit(share) = a + b*price）=====
    share = (df["MOVE"] / df["CUSTOMCOUNT"]).clip(1e-8, 1 - 1e-8)
    y = np.log(share / (1 - share))                     # logit(share)
    X = np.c_[np.ones(len(df)), df["PRICE"].values]     # [1, price]
    # 最小二乘
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = coef
    resid = y - X @ coef
    sigma = resid.std(ddof=1)

    # print(f"Fitted params: alpha={alpha:.4f}, beta={beta:.4f}, sigma={sigma:.4f}")

    # ===== 3. 如果需要，做模拟 + 画图 =====
    if plot:
        import os
        
        # 创建保存图片的文件夹（csv_path 所在目录下的 plots/ 子目录）
        save_dir = os.path.join(os.path.dirname(path), "plots")
        os.makedirs(save_dir, exist_ok=True)

        # 定义一个用模型模拟的函数
        def simulate(prices, markets, seed=42):
            rng = np.random.default_rng(seed)
            prices = np.asarray(prices, float)
            markets = np.asarray(markets, int)
            T = len(prices)
            Q = np.zeros(T, int)
            p = np.zeros(T, float)
            for t in range(T):
                if prices[t] <= 0 or markets[t] <= 0:
                    continue
                V1 = alpha + beta * prices[t] + rng.normal(0, sigma)
                pt = np.exp(V1) / (1 + np.exp(V1))
                pt = float(np.clip(pt, 1e-6, 1 - 1e-6))
                Q[t] = rng.binomial(markets[t], pt)
                p[t] = pt
            return Q, p

        # 用历史价格路径模拟一遍
        Q_sim_hist, _ = simulate(df["PRICE"].values, df["CUSTOMCOUNT"].values)
        df["MOVE_sim"] = Q_sim_hist
        print("历史路径模拟（前几行）：")
        print(df[["WEEK", "PRICE", "MOVE", "MOVE_sim"]].head())

        # 随机生成一段价格路径并模拟 + 画图
        T = 100
        rng = np.random.default_rng(123)
        price_low, price_high = df["PRICE"].min(), df["PRICE"].max()
        prices_rand = rng.uniform(price_low, price_high, size=T)
        markets_rand = np.full(T, int(df["CUSTOMCOUNT"].mean()))
        Q_sim_rand, _ = simulate(prices_rand, markets_rand, seed=123)

        weeks = np.arange(1, T + 1)

        # 图1：随机价格路径
        plt.figure(figsize=(10, 4))
        plt.plot(weeks, prices_rand)
        plt.ylabel("Price")
        plt.xlabel("Week")
        plt.title("Random Price Path")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "random_price_path.png"))
        plt.close()

        # 图2：随机价格下的模拟销量
        plt.figure(figsize=(10, 4))
        plt.plot(weeks, Q_sim_rand)
        plt.ylabel("Sales")
        plt.xlabel("Week")
        plt.title("Simulated Sales (Random Price)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "simulated_sales_random.png"))
        plt.close()

        # 图3：随机价格 vs 模拟销量散点
        plt.figure(figsize=(6, 4))
        plt.scatter(prices_rand, Q_sim_rand)
        plt.xlabel("Price")
        plt.ylabel("Sales")
        plt.title("Price vs Sales (Random Sim)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "price_vs_sales_random.png"))
        plt.close()

        # 图4：真实价格 vs 真实销量散点
        plt.figure(figsize=(6, 4))
        plt.scatter(df["PRICE"], df["MOVE"], alpha=0.6)
        plt.xlabel("Price")
        plt.ylabel("Real Sales (MOVE)")
        plt.title("Real Price vs Real Sales")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "real_price_vs_real_sales.png"))
        plt.close()

        print(f"图像已保存到：{save_dir}")

    # 只返回模型参数（你要的输出）
    return alpha, beta, sigma


# 示例：单独跑这个文件时的用法
if __name__ == "__main__":
    csv_path = "data/dominicks/source_data_processed_filtered/Cheeses/8/26514100000.csv"
    alpha, beta, sigma = fit_logit_demand_model(csv_path, plot=True)
    print("Final params:", alpha, beta, sigma)
