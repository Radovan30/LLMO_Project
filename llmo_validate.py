import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer

# PositionalEncoding pro správné načtení modelu
class PositionalEncoding(Layer):
    def __init__(self, sequence_length, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(sequence_length, d_model)

    @staticmethod
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, sequence_length, d_model):
        angle_rads = self.get_angles(np.arange(sequence_length)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs, **kwargs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "d_model": self.d_model
        })
        return config

# Definice TransformerBlock pro správné načtení modelu
class TransformerBlock(Layer):
    def __init__(self, emb_dim, num_heads, ff_dim, dropout=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout

        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=emb_dim)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(emb_dim)
        ])
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training=False, **kwargs):
        attn_output = self.att(inputs, inputs, training=training, **kwargs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)
        ffn_output = self.ffn(out1, training=training, **kwargs)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.norm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "emb_dim": self.emb_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout_rate,
        })
        return config

# Načtení modelů
repeat_model = load_model("repeat_model.h5", custom_objects={"PositionalEncoding": PositionalEncoding, "TransformerBlock": TransformerBlock})
sort_model = load_model("sort_model.h5", custom_objects={"PositionalEncoding": PositionalEncoding, "TransformerBlock": TransformerBlock})


# Parametry
SEQUENCE_LENGTH = 10
VOCAB_MAX = 101
NUM_TEST_SAMPLES = 1000


# Generování testovacích dat
def generate_test_data(sample_count=NUM_TEST_SAMPLES, sequence_length=SEQUENCE_LENGTH, task="repeat",
                       vocab_max=VOCAB_MAX):
    x, y = [], []
    for _ in range(sample_count):
        seq = np.random.randint(0, vocab_max, sequence_length)
        x.append(seq)
        if task == "repeat":
            y.append(seq.copy())  # Cílová sekvence je stejná jako vstupní
        elif task == "sort":
            y.append(np.sort(seq))  # Cílová sekvence je setříděná verze
    return np.array(x), np.array(y)


# Testovací data
x_test_repeat, y_test_repeat = generate_test_data(task="repeat")
x_test_sort, y_test_sort = generate_test_data(task="sort")


# Funkce pro vyhodnocení modelu
def evaluate_model(model, x_test, y_test, task_name):
    predictions = model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=-1)
    accuracy = np.mean(np.all(predicted_classes == y_test, axis=1)) * 100
    print(f"Model {task_name} Accuracy: {accuracy:.2f}%")

    with open("evaluation_results.txt", "a", encoding="utf-8") as f:
        f.write(f"\nModel {task_name} Accuracy: {accuracy:.2f}%\n")

    # Ukázka prvních 20 výsledků
    for i in range(20):
        sample_input = f"Vstup:      {x_test[i]}"
        sample_pred = f"Predikce:   {predicted_classes[i]}"
        sample_true = f"Očekávané:  {y_test[i]}"
        separator = "-"
        print(sample_input)
        print(sample_pred)
        print(sample_true)
        print(separator)

        with open("evaluation_results.txt", "a", encoding="utf-8") as f:
            f.write(f"{sample_input}\n{sample_pred}\n{sample_true}\n{separator}\n")

    return accuracy, predicted_classes


# Vyhodnocení
accuracy_repeat, predictions_repeat = evaluate_model(repeat_model, x_test_repeat, y_test_repeat, "Repeat")
accuracy_sort, predictions_sort = evaluate_model(sort_model, x_test_sort, y_test_sort, "Sort")


# Grafické vyhodnocení výsledků
def plot_accuracy_bar_chart(accuracies, labels, title, filename):
    plt.figure(figsize=(6, 4))
    plt.bar(labels, accuracies, color=['blue', 'orange'])
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 1, f"{acc:.2f}%", ha='center', fontsize=12)
    plt.savefig(filename)
    plt.close()


# Vykreslení grafu přesnosti
plot_accuracy_bar_chart([accuracy_repeat, accuracy_sort], ["Repeat Task", "Sort Task"], "Model Accuracy Comparison",
                        "model_accuracy_comparison.png")

print(f"Final Accuracy - Repeat Task: {accuracy_repeat:.2f}%")
print(f"Final Accuracy - Sort Task: {accuracy_sort:.2f}%")