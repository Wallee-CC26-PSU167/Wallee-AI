import numpy as np
import joblib
import tensorflow as tf

from tensorflow.keras.models import load_model
from typing import List, Dict, Any

class FeatureNoiseLayer(tf.keras.layers.Layer):

    def __init__(self, stddev: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev

    def call(self, inputs, training=False):
        if training:
            return inputs + tf.random.normal(
                tf.shape(inputs),
                stddev=self.stddev
            )
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({"stddev": self.stddev})
        return config

class AnomalyService:

    def __init__(
        self,
        model_path,
        scaler_path,
        features_path,
        threshold_path
    ):

        self.model = load_model(
            model_path,
            custom_objects={
                "FeatureNoiseLayer": FeatureNoiseLayer
            }
        )

        self.scaler = joblib.load(scaler_path)
        self.features = joblib.load(features_path)
        self.threshold = joblib.load(threshold_path)

        self.p95 = self.threshold["p95"]
        self.p99 = self.threshold["p99"]

        print(
            f"AnomalyService ready. "
            f"Features={len(self.features)} "
            f"p95={self.p95:.6f} "
            f"p99={self.p99:.6f}"
        )

    # =====================================================
    # AUTOENCODER
    # =====================================================

    def _autoencoder_score(
        self,
        enriched: Dict[str, Any]
    ) -> float:

        row = {col: 0.0 for col in self.features}

        for col in self.features:

            value = enriched.get(col)

            if value is None:
                continue

            try:
                row[col] = float(value)
            except Exception:
                pass

        X = np.array(
            [[row[c] for c in self.features]],
            dtype=np.float32
        )

        X_scaled = self.scaler.transform(X)

        reconstructed = self.model.predict(
            X_scaled,
            verbose=0
        )

        score = float(
            np.mean(
                np.square(
                    X_scaled - reconstructed
                )
            )
        )

        return score

    # =====================================================
    # MAIN DETECTION
    # =====================================================

    def detect(
        self,
        transaction_id: str,
        enriched: Dict[str, Any]
    ) -> List[Dict[str, Any]]:

        anomalies = []
        anomaly_counter = 1

        # =================================================
        # DERIVED FEATURES
        # =================================================

        amount = enriched.get("amount", 0)

        historical_avg_amount = enriched.get(
            "historical_avg_amount"
        )

        amount_ratio = 0

        if (
            historical_avg_amount is not None
            and historical_avg_amount > 0
        ):
            amount_ratio = (
                amount /
                historical_avg_amount
            )

        merchant_monthly_freq = enriched.get(
            "merchant_monthly_freq",
            0
        )

        merchant_avg_freq = (
            enriched.get("merchant_avg_freq")
            or 0
        )

        merchant_freq_ratio = 0

        if merchant_avg_freq > 0:
            merchant_freq_ratio = (
                merchant_monthly_freq /
                merchant_avg_freq
            )

        trx_time = enriched.get("time")

        usual_hour_min = enriched.get(
            "usual_hour_min"
        )

        usual_hour_max = enriched.get(
            "usual_hour_max"
        )

        is_unusual_hour = False

        if (
            trx_time
            and usual_hour_min is not None
            and usual_hour_max is not None
        ):
            try:

                hour = int(
                    trx_time.split(":")[0]
                )

                is_unusual_hour = (
                    hour < usual_hour_min
                    or hour > usual_hour_max
                )

            except Exception:
                pass

        # =================================================
        # SPENDING SPIKE
        # =================================================

        if amount_ratio > 3:

            anomalies.append(
                {
                    "id": f"an_{anomaly_counter:03d}",
                    "type": "SPENDING_SPIKE",
                    "message": "Pengeluaran lebih tinggi dari biasanya",
                    "detail": [
                        {
                            "historical_avg_amount":
                                round(historical_avg_amount, 2)
                                if historical_avg_amount is not None
                                else 0,
                            "current_amount":
                                amount,
                            "amount_ratio":
                                round(
                                    amount_ratio,
                                    2
                                )
                        }
                    ],
                    "dismissed": False
                }
            )

            anomaly_counter += 1

        # =================================================
        # UNUSUAL TIME
        # =================================================

        if is_unusual_hour:

            anomalies.append(
                {
                    "id": f"an_{anomaly_counter:03d}",
                    "type": "UNUSUAL_TIME",
                    "message": "Transaksi dilakukan di luar jam biasanya",
                    "detail": [
                        {
                            "time":
                                trx_time,
                            "usual_hour_min":
                                usual_hour_min,
                            "usual_hour_max":
                                usual_hour_max
                        }
                    ],
                    "dismissed": False
                }
            )

            anomaly_counter += 1

        # =================================================
        # FREQUENCY SPIKE
        # =================================================

        if merchant_freq_ratio > 3:

            anomalies.append(
                {
                    "id": f"an_{anomaly_counter:03d}",
                    "type": "FREQUENCY_SPIKE",
                    "message": "Frekuensi transaksi merchant meningkat tajam",
                    "detail": [
                        {
                            "merchant":
                                enriched.get(
                                    "merchant",
                                    ""
                                ),
                            "merchant_monthly_freq":
                                merchant_monthly_freq,
                            "merchant_avg_freq":
                                round(
                                    merchant_avg_freq,
                                    2
                                ),
                            "merchant_freq_ratio":
                                round(
                                    merchant_freq_ratio,
                                    2
                                )
                        }
                    ],
                    "dismissed": False
                }
            )

            anomaly_counter += 1

        # =================================================
        # PRICE SPIKE
        # =================================================

        for item in enriched.get(
            "items",
            []
        ):

            usual_price = item.get(
                "usual_price"
            )

            current_price = item.get(
                "harga"
            )

            if (
                usual_price is None
                or usual_price <= 0
            ):
                continue

            ratio = (
                current_price /
                usual_price
            )

            if ratio > 3:

                anomalies.append(
                    {
                        "id": f"an_{anomaly_counter:03d}",
                        "type": "PRICE_SPIKE",
                        "message": "Harga item jauh lebih tinggi dari biasanya",
                        "detail": [
                            {
                                "item_name":
                                    item.get(
                                        "item_name"
                                    ),
                                "harga":
                                    current_price,
                                "usual_price":
                                    round(
                                        usual_price,
                                        2
                                    ),
                                "price_ratio":
                                    round(
                                        ratio,
                                        2
                                    )
                            }
                        ],
                        "dismissed": False
                    }
                )

                anomaly_counter += 1

        # =================================================
        # AUTOENCODER
        # =================================================

        score = self._autoencoder_score(
            enriched
        )

        if (
            score > self.p95
            and len(anomalies) == 0
        ):

            anomalies.append(
                {
                    "id": f"an_{anomaly_counter:03d}",
                    "type": "AUTOENCODER_ANOMALY",
                    "message": "Pola transaksi tidak biasa",
                    "detail": [
                        {
                            "anomaly_score":
                                round(
                                    score,
                                    8
                                ),
                            "threshold":
                                round(
                                    self.p95,
                                    8
                                )
                        }
                    ],
                    "dismissed": False
                }
            )

        return anomalies