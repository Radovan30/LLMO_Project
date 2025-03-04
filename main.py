import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Dense, Embedding, Input
from tensorflow.keras.models import Model

# Globální parametry
NUM_SAMPLES = 50000
SEQUENCE_LENGTH = 10
VOCAB_MAX = 101  # Čísla 0-100

# Funkce pro generování datasetu
def generate_data(sample_count=NUM_SAMPLES, sequence_length=SEQUENCE_LENGTH, task="repeat", vocab_max=VOCAB_MAX):
    x, y = [], []
    for _ in range(sample_count):
        seq = np.random.randint(0, vocab_max, sequence_length)
        x.append(seq)
        if task == "repeat":
            y.append(seq)  # Cílová sekvence je stejná jako vstupní
        elif task == "sort":
            y.append(np.sort(seq))  # Cílová sekvence je setříděná verze
    return np.array(x), np.array(y)

# Vytvoření datasetů pro obě úlohy
x_repeat, y_repeat = generate_data(task="repeat")
x_sort, y_sort = generate_data(task="sort")

# Funkce pro vytvoření datasetu
def create_tf_dataset(x, y, batch_size=64):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset_repeat = create_tf_dataset(x_repeat, y_repeat)
train_dataset_sort = create_tf_dataset(x_sort, y_sort)

# Implementace Positional Encoding vrstvy
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, d_model):
        super(PositionalEncoding, self).__init__()
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
        # Použijeme sin na sudé indexy a cos na liché indexy
        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]  # tvar (1, sequence_length, d_model)
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs, **kwargs):
        # inputs: (batch_size, sequence_length, d_model)
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "d_model": self.d_model
        })
        return config

# Definice Transformer bloku s dropoutem
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, emb_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout

        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=emb_dim)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(emb_dim)
        ])
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

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

# Funkce pro sestavení Transformer modelu
def build_transformer_model(sequence_length=SEQUENCE_LENGTH, vocab_size=VOCAB_MAX, emb_dim=128, num_heads=4, ff_dim=512, num_layers=1):
    inputs = Input(shape=(sequence_length,))
    x = Embedding(vocab_size, emb_dim)(inputs)
    # Přidání Positional Encoding
    x = PositionalEncoding(sequence_length, emb_dim)(x)
    for _ in range(num_layers):
        x = TransformerBlock(emb_dim, num_heads, ff_dim)(x)
    x = Dense(vocab_size, activation="softmax")(x)

    model = Model(inputs, x)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

# Callbacky pro trénink
early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
csv_logger = CSVLogger("training_log.csv", append=True)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)

# Vlastní callback pro ukončení tréninku, pokud model 5× dosáhne accuracy 1.0
class EarlyStoppingByAccuracy(tf.keras.callbacks.Callback):
    def __init__(self, patience=5, threshold=1.0):
        super(EarlyStoppingByAccuracy, self).__init__()
        self.patience = patience
        self.threshold = threshold
        self.counter = 0

    def on_epoch_end(self, epoch, logs=None):
        current_acc = logs.get("accuracy")
        if current_acc is not None and current_acc >= self.threshold:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"\nModel dosáhl accuracy {self.threshold} {self.patience}x po sobě. Ukončuji trénink.")
                self.model.stop_training = True
        else:
            self.counter = 0

early_stopping_acc = EarlyStoppingByAccuracy(patience=5, threshold=1.0)

# Funkce pro vykreslení tréninkové historie
def plot_training_history(history, filename):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Hodnota')
    plt.legend()
    plt.title('Trénovací křivka')
    plt.savefig(filename)
    plt.close()

# Funkce pro trénování a uložení modelu a grafu
def train_and_save_model(model, dataset, epochs, model_filename, plot_filename):
    callbacks_list = [early_stopping, csv_logger, reduce_lr, early_stopping_acc]
    history = model.fit(dataset, epochs=epochs, callbacks=callbacks_list)
    model.save(model_filename)
    plot_training_history(history, plot_filename)
    return history

# Trénování modelu pro opakování sekvence
repeat_model = build_transformer_model(sequence_length=SEQUENCE_LENGTH, vocab_size=VOCAB_MAX, emb_dim=128, num_heads=4, ff_dim=512, num_layers=1)
repeat_model.summary()
history_repeat = train_and_save_model(repeat_model, train_dataset_repeat, epochs=15, model_filename="repeat_model.h5", plot_filename="repeat_training_plot.png")

# Trénování modelu pro řazení sekvence (s více Transformer bloky)
sort_model = build_transformer_model(sequence_length=SEQUENCE_LENGTH, vocab_size=VOCAB_MAX, emb_dim=128, num_heads=4, ff_dim=512, num_layers=2)
sort_model.summary()
history_sort = train_and_save_model(sort_model, train_dataset_sort, epochs=1000, model_filename="sort_model.h5", plot_filename="sort_training_plot.png")
