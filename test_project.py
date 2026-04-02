import unittest
import numpy as np
import pandas as pd
import os
import Main_Project_code as proj


class Tests(unittest.TestCase):

    # Disables plots
    def setUp(self):
        proj.plt.show = lambda: None
        proj.plot_time = lambda: None

    # CSV TESTS

    # basic test
    def test_load_csv_simple(self):
        df = pd.DataFrame({
            "time": np.linspace(0, 1, 100),
            "signal": np.sin(np.linspace(0, 10, 100))
        })

        file = "temp_test.csv"
        df.to_csv(file, index=False)

        proj.filedialog.askopenfilename = lambda **kwargs: file

        proj.load_csv()

        self.assertEqual(len(proj.time), 100)
        self.assertEqual(len(proj.signal), 100)
        self.assertTrue(proj.fs > 0)

        os.remove(file)

    # test of missing values
    def test_load_csv_missing_vals(self):
        df = pd.DataFrame({
            "time": np.linspace(0, 1, 50),
            "signal": [1, np.nan, 3, np.nan, 5] * 10
        })

        file = "temp_nan.csv"
        df.to_csv(file, index=False)

        proj.filedialog.askopenfilename = lambda **kwargs: file

        proj.load_csv()

        self.assertFalse(np.isnan(proj.signal).any())

        os.remove(file)

    # test of alternative column names
    def test_load_csv_alt_columns(self):
        df = pd.DataFrame({
            "t": np.linspace(0, 1, 100),
            "lead_I": np.sin(np.linspace(0, 10, 100))
        })

        file = "temp_alt.csv"
        df.to_csv(file, index=False)

        proj.filedialog.askopenfilename = lambda **kwargs: file

        proj.load_csv()

        self.assertEqual(len(proj.time), 100)
        self.assertEqual(len(proj.signal), 100)
        self.assertTrue(proj.fs > 0)

        os.remove(file)


    # FILTER TESTS

    # test of lowpass
    def test_butter_lowpass(self):
        fs = 100
        t = np.linspace(0, 1, fs)
        sig = np.sin(2*np.pi*2*t) + np.sin(2*np.pi*40*t)

        filtered = proj.butter_filter(sig, 10, fs, "low")

        self.assertEqual(len(filtered), len(sig))
        self.assertTrue(np.std(filtered) < np.std(sig))

    def test_butter_highpass(self):
        fs = 100
        t = np.linspace(0, 1, fs)
        sig = np.sin(2*np.pi*1*t) + np.sin(2*np.pi*20*t)

        filtered = proj.butter_filter(sig, 5, fs, "high")

        self.assertEqual(len(filtered), len(sig))

    def test_butter_bandpass(self):
        fs = 100
        t = np.linspace(0, 1, fs)
        sig = np.sin(2*np.pi*1*t) + np.sin(2*np.pi*20*t)

        filtered = proj.butter_filter(sig, [5, 30], fs, "band")

        self.assertEqual(len(filtered), len(sig))
        

    # ---------------------------
    # APPLY FILTER FUNCTIONS
    # ---------------------------
    def test_apply_lpf(self):
        proj.signal = np.random.randn(100)
        proj.fs = 100

        proj.apply_lpf()

        self.assertEqual(len(proj.processed_signal), 100)

    def test_apply_hpf(self):
        proj.signal = np.random.randn(100)
        proj.fs = 100

        proj.apply_hpf()

        self.assertEqual(len(proj.processed_signal), 100)

    def test_apply_bpf(self):
        proj.signal = np.random.randn(100)
        proj.fs = 100

        proj.apply_bpf()

        self.assertEqual(len(proj.processed_signal), 100)

    # ---------------------------
    # STATISTICS
    # ---------------------------
    def test_compute_stats(self):
        proj.processed_signal = np.array([1, 2, 3, 4])

        class DummyVar:
            def set(self, value):
                self.value = value

        proj.stats_text = DummyVar()

        proj.compute_stats()

        self.assertIn("Mean", proj.stats_text.value)
        self.assertIn("STD", proj.stats_text.value)
        self.assertIn("RMS", proj.stats_text.value)
        self.assertIn("Peak-Peak", proj.stats_text.value)

    # ---------------------------
    # FFT
    # ---------------------------
    def test_show_fft(self):
        fs = 100
        t = np.linspace(0, 1, fs)
        sig = np.sin(2*np.pi*10*t)

        proj.processed_signal = sig
        proj.fs = fs

        proj.show_fft()  # should not crash

    # ---------------------------
    # RESET
    # ---------------------------
    def test_reset_signal(self):
        proj.signal = np.array([1, 2, 3])
        proj.processed_signal = np.array([9, 9, 9])

        proj.reset_signal()

        self.assertTrue(np.array_equal(proj.signal, proj.processed_signal))


# ---------------------------
# RUN TESTS
# ---------------------------
if __name__ == "__main__":
    unittest.main()