import unittest
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')

import Main_project_code_FV as proj


class Tests(unittest.TestCase):

    def setUp(self):
        # Disable plots
        proj.plt.show = lambda: None
        proj.plot_time = lambda: None

        class DummyVar:
            def __init__(self, value=None):
                self.value = value

            def get(self):
                return self.value

            def set(self, value):
                self.value = value

        proj.signal_type = DummyVar("ECG")
        proj.filter_method = DummyVar("IIR")
        proj.stats_text = DummyVar()

    # ---------------------------
    # CSV TESTS
    # ---------------------------
    def test_load_csv_simple(self):
        df = pd.DataFrame({
            "time": np.linspace(0, 1, 100),
            "signal": np.sin(np.linspace(0, 10, 100))
        })

        file = "temp.csv"
        df.to_csv(file, index=False)

        proj.filedialog.askopenfilename = lambda **k: file
        proj.load_csv()

        self.assertEqual(len(proj.time), 100)
        self.assertEqual(len(proj.signal), 100)
        self.assertTrue(proj.fs > 0)

        os.remove(file)

    def test_load_csv_missing_vals(self):
        df = pd.DataFrame({
            "time": np.linspace(0, 1, 50),
            "signal": [1, np.nan, 3, np.nan, 5] * 10
        })

        file = "temp_nan.csv"
        df.to_csv(file, index=False)

        proj.filedialog.askopenfilename = lambda **k: file
        proj.load_csv()

        self.assertTrue(np.isfinite(proj.signal).all())

        os.remove(file)

    def test_load_csv_alt_columns(self):
        df = pd.DataFrame({
            "t": np.linspace(0, 1, 100),
            "lead_I": np.sin(np.linspace(0, 10, 100))
        })

        file = "temp_alt.csv"
        df.to_csv(file, index=False)

        proj.filedialog.askopenfilename = lambda **k: file
        proj.load_csv()

        self.assertEqual(len(proj.time), 100)
        self.assertEqual(len(proj.signal), 100)

        os.remove(file)

    # ---------------------------
    # FILTER TEST
    # ---------------------------
    def test_butter_lowpass(self):
        fs = 100
        t = np.linspace(0, 1, fs)
        sig = np.sin(2*np.pi*2*t) + np.sin(2*np.pi*40*t)

        filtered = proj.butter_filter(sig, 10, fs, "low")

        self.assertEqual(len(filtered), len(sig))
        self.assertTrue(np.std(filtered) < np.std(sig))

    # ---------------------------
    # STATS TEST
    # ---------------------------
    def test_compute_stats(self):
        proj.processed_signal = np.array([1,2,3,4])

        proj.fs = 10
        proj.time = np.linspace(0, 1, 4)

        proj.compute_stats()

        output = proj.stats_text.value

        self.assertIn("Mean: 2.500", output)
        self.assertIn("STD:", output)
        self.assertIn("RMS:", output)
        self.assertIn("P2P: 3.000", output)
    # ---------------------------
    # RESET TEST
    # ---------------------------
    def test_reset_signal(self):
        proj.original_signal = np.array([1,2,3])
        proj.processed_signal = np.array([9,9,9])
        proj.time = np.linspace(0,1,3)

        proj.reset_signal()

        self.assertTrue(np.array_equal(proj.original_signal, proj.processed_signal))


if __name__ == "__main__":
    unittest.main()