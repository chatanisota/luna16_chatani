"""
    アンバランスデータ解消のためのライブラリ
    https://blog.amedama.jp/entry/under-bagging-kfold
"""
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import numpy as np

# アンバランスデータの解消
# UnderSampling + Bagging
class UnderBaggingKFold(BaseCrossValidator):
    """CV に使うだけで UnderBagging できる KFold 実装

    NOTE: 少ないクラスのデータは各 Fold で重複して選択される"""

    def __init__(self, n_splits=10, shuffle=True, random_states=None,
                 test_size=0.2, whole_testing=False):
        """
        :param n_splits: Fold の分割数
        :param shuffle: 分割時にデータをシャッフルするか
        :param random_states: 各 Fold の乱数シード
        :param test_size: Under-sampling された中でテスト用データとして使う割合
        :param whole_testing: Under-sampling で選ばれなかった全てのデータをテスト用データに追加するか
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_states = random_states
        self.test_size = test_size
        self.whole_testing = whole_testing

        if random_states is not None:
            # 各 Fold の乱数シードが指定されているなら分割数をそれに合わせる
            self.n_splits = len(random_states)
        else:
            # 乱数シードが指定されていないときは分割数だけ None で埋めておく
            self.random_states = [None] * self.n_splits

        # 分割数だけ Under-sampling 用のインスタンスを作っておく
        self.samplers_ = [
            RandomUnderSampler(random_state=random_state)
            for random_state in self.random_states
        ]

    def split(self, X, y=None, groups=None):
        """データを学習用とテスト用に分割する"""
        if X.ndim < 2:
            # RandomUnderSampler#fit_resample() は X が 1d-array だと文句を言う
            X = np.vstack(X)

        for i in range(self.n_splits):
            # データを Under-sampling して均衡データにする
            sampler = self.samplers_[i]
            _, y_sampled = sampler.fit_resample(X, y)
            # 選ばれたデータのインデックスを取り出す
            sampled_indices = sampler.sample_indices_

            # 選ばれたデータを学習用とテスト用に分割する
            split_data = train_test_split(sampled_indices,
                                          shuffle=self.shuffle,
                                          test_size=self.test_size,
                                          stratify=y_sampled,
                                          random_state=self.random_states[i],
                                          )
            train_indices, test_indices = split_data

            if self.whole_testing:
                # Under-sampling で選ばれなかったデータをテスト用に追加する
                mask = np.ones(len(X), dtype=np.bool)
                mask[sampled_indices] = False
                X_indices = np.arange(len(X))
                non_sampled_indices = X_indices[mask]
                test_indices = np.concatenate([test_indices,
                                               non_sampled_indices])

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
