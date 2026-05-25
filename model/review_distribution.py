import math
import random

# 全局基础分布（用你给的数据）
BASE_COUNTS = {1: 21, 2: 9, 3: 6, 4: 14, 5: 50}
TOTAL = sum(BASE_COUNTS.values())
BASE_PROBS = {k: v / TOTAL for k, v in BASE_COUNTS.items()}

def get_tilted_probs(target_mean, tol=1e-4, max_iter=50):
    """
    根据指定的“目标平均分”生成符合该平均分的评分概率分布
    方法：指数倾斜 (Exponential Tilting)
    
    ✨原理解释：
    ================================
    已知全局评分分布 BASE_PROBS：
    p(1)=0.21, ... p(5)=0.40
    
    如果我们希望某个产品的最终分布 q(k) 满足：
        期望 E[k] = target_mean （比如 3.6）
    
    我们不要凭空捏造一个新分布
    而是 **基于全局分布做最小变化的调整（最自然的改变）**
    
    数学形式：
        q(k; θ) ∝ p(k) * exp(θ * k)
    
    含义：
    - exp(θ*k) 是一个“偏置因子”
    - θ > 0：拉向高分（因为高 k 时权重大）
    - θ < 0：拉向低分
    - θ = 0：完全不变，q(k)=p(k)
    
    我们通过调整 θ，使得：
        Sum( k * q(k;θ) ) ≈ target_mean
    
    这是信息论中“最小信息扰动”的方法
    能最大限度保留原评分分布形状（高分多的趋势不变）
    
    实现方式：
    由于 E[k] 与 θ 不是线性关系
    我们使用 **二分搜索** 找到满足条件的 θ
    ================================
    """
    # 边界保护：分数只在 1～5
    target_mean = max(1.0, min(5.0, target_mean))
    
    def mean_given_theta(theta):
        # 计算在当前 theta 下的概率和期望
        weights = {}
        for k, p in BASE_PROBS.items():
            weights[k] = p * math.exp(theta * k)
        z = sum(weights.values())
        probs = {k: w / z for k, w in weights.items()}
        mean = sum(k * probs[k] for k in probs)
        return mean, probs

    # 二分搜索 theta
    # theta_low 往低分倾斜，theta_high 往高分倾斜
    theta_low, theta_high = -10.0, 10.0
    best_probs = BASE_PROBS

    for _ in range(max_iter):
        theta_mid = (theta_low + theta_high) / 2
        mean_mid, probs_mid = mean_given_theta(theta_mid)
        best_probs = probs_mid

        if abs(mean_mid - target_mean) < tol:
            break

        if mean_mid < target_mean:
            # 均分太低，说明高分不够，theta 要更大一点
            theta_low = theta_mid
        else:
            # 均分太高，说明高分太多，theta 要小一点
            theta_high = theta_mid
    
    return best_probs

def sample_rating_for_product(target_mean):
    """
    给定某个产品的平均分 target_mean，
    按构造出的分布随机采样一条评分（1~5）
    """
    probs = get_tilted_probs(target_mean)
    r = random.random()
    cum = 0.0
    for k in sorted(probs.keys()):
        cum += probs[k]
        if r <= cum:
            return k

# 示例：看一下 3.6 的分布 & 随机采样
if __name__ == "__main__":
    probs_36 = get_tilted_probs(4.6)
    print("target_mean = 4.6 时的概率分布：", probs_36)

    # 简单检验一下期望
    approx_mean = sum(k * p for k, p in probs_36.items())
    print("近似期望：", approx_mean)

    # 随机采样 10 次
    samples = [sample_rating_for_product(4.6) for _ in range(50)]
    print("10 次采样结果：", samples)