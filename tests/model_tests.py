from taxus import GP
from taxus.likelihoods import LIKELIHOODS
import pandas as pd
from unittest import TestCase


class ModelTests(TestCase):
    def setUp(self):
        self.train_x_df = pd.DataFrame(
            dict(
                a=[1., 2., 2.],
                b=[3., 4., 2.]
            )
        )

        self.train_y_df = pd.DataFrame(
            dict(
                d=[10., 15., 20.],
                e=[30., 14., 21.],
            )
        ).T

        self.test_df = pd.DataFrame(
            dict(
                a=[1., 2.5, 2.6, 3.1],
                b=[0.1, 0.2, 4.1, 4.2]
            )
        )

    def test_simple_init(self):
        train_y_df = pd.DataFrame(self.train_y_df.loc['d'])
        GP('~ a + b', self.train_x_df, train_y_df)

    def test_fit(self):
        train_y_df = pd.DataFrame(self.train_y_df.loc['d'])
        gp = GP('~ a + b', self.train_x_df, train_y_df)
        loss = gp.fit(n_steps=2, debug=True)
        self.assertTrue(type(loss) == float)

    def test_predict(self):
        train_y_df = pd.DataFrame(self.train_y_df.loc['d'])
        gp = GP('~ a + b', self.train_x_df, train_y_df)
        gp.predict(self.test_df)

    def test_likelihoods(self):
        train_y_df = pd.DataFrame(self.train_y_df.loc['d'])
        for likelihood in LIKELIHOODS.keys():
            gp = GP('~ a + b', self.train_x_df, train_y_df,
                    likelihood=likelihood)
            gp.fit(n_steps=2)

    def test_kernels(self):
        train_y_df = pd.DataFrame(self.train_y_df.loc['d'])
        for kernel in ('rbf', 'linear'):
            gp = GP('~ a + b', self.train_x_df, train_y_df, kernel=kernel)
            gp.fit(n_steps=2)
