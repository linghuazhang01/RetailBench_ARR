import json
import random
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import asdict, dataclass

from util.file import dump_json, load_json


# ================== 1. 原始分段函数 ==================
def rating_gain(r, star_point: float = 4.0, X: float = 0.70):
    """
    原始分段函数：评分 r ∈ [0, 5] → 相对销量提升比例 G(r)

    参数
    ----
    r : float 或 np.ndarray
        评分（0~5）
    star_point : float
        从哪个星级开始乘 (X - 0.24)，例如 4.0
    X : float
        4→5 星总的增长量，例如 0.70 表示 70% 增长
    """
    r = np.asarray(r)
    return (
        0.04 * np.maximum(r - 2, 0) +
        0.20 * np.maximum(r - 3, 0) +
        (X - 0.24) * np.maximum(r - star_point, 0)
    )


# ================== 2. 拟合结果的数据结构 ==================
@dataclass
class SmoothFitResult:
    """
    存放平滑拟合结果和误差指标
    """
    # 多项式系数（从最高次到常数项）
    poly_coeffs: np.ndarray

    # 最大绝对误差及其位置
    max_err: float
    max_err_r: float
    max_err_true: float
    max_err_pred: float

    # 整体误差指标
    mse: float
    rmse: float
    r2: float

    def to_json(self) -> str:
        """
        转换为 JSON 字符串（自动把 numpy.ndarray 转为 list）
        """
        data = asdict(self)
        data["poly_coeffs"] = self.poly_coeffs.tolist()  # ndarray → list
        return data

    


# ================== 3. 使用多项式拟合并评估的封装函数 ==================
def fit_smooth_gain(
    star_point: float = 4.0,
    X: float = 0.70,
    degree: int = 4,
    r_min: float = 0.0,
    r_max: float = 5.0,
    num_samples: int = 501,
    plot: bool = False,
) -> SmoothFitResult:
    """
    对 rating_gain 进行多项式拟合，并返回拟合结果与误差评估。

    参数
    ----
    star_point : float
        从哪个星级开始乘 (X - 0.24)
    X : float
        4→5 星总的增长量，例如 0.70 表示 70% 增长
    degree : int
        多项式拟合阶数
    r_min, r_max : float
        拟合区间的评分范围（默认 0~5）
    num_samples : int
        用于拟合的采样点个数
    plot : bool
        是否画出原始函数、拟合曲线和误差

    返回
    ----
    SmoothFitResult
        包含多项式系数和各种误差指标
    """
    # 1) 生成采样点
    r_vals = np.linspace(r_min, r_max, num_samples)
    y_vals = rating_gain(r_vals, star_point=star_point, X=X)

    # 2) 多项式拟合
    poly_coeffs = np.polyfit(r_vals, y_vals, deg=degree)

    # 用得到的系数构造平滑函数
    smooth_vals = np.polyval(poly_coeffs, r_vals)

    # 3) 误差评估
    abs_err = np.abs(smooth_vals - y_vals)
    max_err = float(abs_err.max())
    max_err_idx = int(abs_err.argmax())
    max_err_r = float(r_vals[max_err_idx])
    max_err_true = float(y_vals[max_err_idx])
    max_err_pred = float(smooth_vals[max_err_idx])

    mse = float(np.mean((smooth_vals - y_vals) ** 2))
    rmse = float(np.sqrt(mse))

    ss_res = float(np.sum((smooth_vals - y_vals) ** 2))
    ss_tot = float(np.sum((y_vals - y_vals.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot)

    # 4) 可选画图
    if plot:
        # 原函数 vs 拟合函数
        plt.figure()
        # plt.plot(r_vals, y_vals, label="Original G(r)")
        plt.plot(r_vals, smooth_vals, linestyle="--", label="Smooth Poly G_smooth(r)")
        # plt.scatter([max_err_r], [max_err_true], marker="o", label="Max Error Point")
        plt.xlabel("Rating r")
        plt.ylabel("Gain")
        plt.title(f"Rating Gain vs Polynomial Fit (degree={degree})")
        plt.legend()
        plt.grid(True)

        # 误差曲线
        plt.figure()
        plt.plot(r_vals, abs_err)
        plt.xlabel("Rating r")
        plt.ylabel("Absolute Error |G_smooth - G|")
        plt.title("Absolute Error of Polynomial Fit")
        plt.grid(True)

        plt.show()

    # 5) 打包结果
    result = SmoothFitResult(
        poly_coeffs=poly_coeffs,
        max_err=max_err,
        max_err_r=max_err_r,
        max_err_true=max_err_true,
        max_err_pred=max_err_pred,
        mse=mse,
        rmse=rmse,
        r2=r2,
    )
    return result


# ================== 4. 给外部用的平滑函数封装 ==================
def smooth_gain(r, poly_coeffs):
    """
    使用拟合得到的多项式系数，计算平滑后的 G_smooth(r)。

    参数
    ----
    r : float 或 np.ndarray
        评分
    poly_coeffs : np.ndarray
        np.polyfit 得到的系数数组（从高次到常数项）

    返回
    ----
    np.ndarray 或 float
        对应的平滑提升比例
    """
    r = np.asarray(r)
    return np.polyval(poly_coeffs, r)


import json
import numpy as np


class ReviewStarSmoothModel:
    """
    评分 → 平滑增益 G(r) 模型
    从 JSON 文件加载 poly_coeffs，对评分进行拟合后的增益计算
    """

    def __init__(self, json_path: str):
        """
        加载平滑参数文件
        """
        with open(json_path, "r", encoding="utf-8") as f:
            self.params = json.load(f)

    @staticmethod
    def simulate_ratings(target_mean: float | None, n: int = 1) -> list[int]:
        """
        Sample star ratings (1-5) so that the expected mean approaches target_mean.
        Uses exponential tilting over a base distribution (see review_distribution).
        """
        from model.review_distribution import get_tilted_probs

        if n <= 0:
            return []

        try:
            mean_val = float(target_mean) if target_mean is not None else 3.6
        except (TypeError, ValueError):
            mean_val = 3.6

        # Clamp to valid rating range
        mean_val = max(1.0, min(5.0, mean_val))

        probs = get_tilted_probs(mean_val)
        keys = sorted(probs.keys())
        thresholds: list[tuple[float, int]] = []
        total = 0.0
        for k in keys:
            total += float(probs[k])
            thresholds.append((total, int(k)))

        ratings: list[int] = []
        for _ in range(n):
            r = random.random()
            picked = thresholds[-1][1]
            for th, k in thresholds:
                if r <= th:
                    picked = k
                    break
            ratings.append(picked)

        return ratings

    def predict(self, category: str, rating):
        """
        根据 category 与 rating 计算平滑增益 G_smooth(r)

        参数：
        - category: 文件中对应的 key，如 "Beer" / "Cookies"
        - rating: 数值或 numpy 数组

        返回：
        - 平滑增益（标量或 numpy 数组）
        """

        if category not in self.params:
            raise KeyError(f"Category '{category}' not found in smooth parameter file.")

        coeffs = np.asarray(self.params[category]["poly_coeffs"])
        rating = np.asarray(rating)

        return np.polyval(coeffs, rating)



if __name__ == "__main__":

    # review_begin_effect_point = load_json('data/review/begin_effect_point.json')

    # review_effect_power = load_json('data/review/effect_power.json')


    # parameter = {

    # }

    # for key, value in review_effect_power.items():
    #     parameter[key] = fit_smooth_gain(
    #         star_point=review_begin_effect_point[key],
    #         X=value / 100,
    #         degree=4,
    #         plot=True
    #     ).to_json()

    # dump_json(parameter, 'model/review_star_smooth_params.json')

    review_model = ReviewStarSmoothModel('data/still/review/review_star_smooth_params.json')

    print(review_model.predict('Bath Soap', 3.5))

    print(review_model.predict('Bath Soap', 4.0))

    # -------------------- tests for simulate_ratings --------------------
    def _mean(vals: list[int]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    def _test_simulate_ratings_basic() -> None:
        random.seed(0)
        ratings = ReviewStarSmoothModel.simulate_ratings(4.2, n=1000)
        print(f"Simulated ratings (mean={_mean(ratings):.2f}): {ratings[:20]}...")
        assert len(ratings) == 1000
        assert all(1 <= r <= 5 for r in ratings)
        avg = _mean(ratings)
        assert 3.7 <= avg <= 4.7, f"mean out of expected range: {avg}"

    def _test_simulate_ratings_monotonic() -> None:
        random.seed(1)
        low = ReviewStarSmoothModel.simulate_ratings(2.2, n=2000)
        random.seed(1)
        high = ReviewStarSmoothModel.simulate_ratings(4.4, n=2000)
        assert _mean(high) > _mean(low)

    def _test_simulate_ratings_default() -> None:
        random.seed(2)
        ratings = ReviewStarSmoothModel.simulate_ratings(None, n=1000)
        avg = _mean(ratings)
        assert 3.1 <= avg <= 4.1, f"default mean out of expected range: {avg}"

    def _test_simulate_ratings_zero() -> None:
        assert ReviewStarSmoothModel.simulate_ratings(4.2, n=0) == []

    print("Running simulate_ratings tests...")
    _test_simulate_ratings_basic()
    _test_simulate_ratings_monotonic()
    _test_simulate_ratings_default()
    _test_simulate_ratings_zero()
    print("simulate_ratings tests passed.")
