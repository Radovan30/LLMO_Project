import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, MultiHeadAttention, LayerNormalization, Dropout, Dense
from tensorflow.keras.models import load_model

# Funkce pro kontrolu, zda uživatel zadal "exit"
def check_exit(user_input):
    if user_input.strip().lower() == "exit":
        print("Ukončuji program.")
        return True
    return False

# Definice vlastní vrstvy PositionalEncoding
class PositionalEncoding(Layer):
    def __init__(self, sequence_length, d_model, **kwargs):
        super().__init__(**kwargs)
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
        pos_encoding = pos_encoding[np.newaxis, ...]  # tvar (1, sequence_length, d_model)
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs, **kwargs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "d_model": self.d_model
        })
        return config

# Definice vlastní vrstvy TransformerBlock
class TransformerBlock(Layer):
    def __init__(self, emb_dim, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=emb_dim)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(emb_dim),
        ])
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs, training=False, **kwargs):
        attn_output = self.att(inputs, inputs, training=training)
        out1 = self.norm1(inputs + attn_output)
        ffn_output = self.ffn(out1, training=training)
        return self.norm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "emb_dim": self.emb_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout,
        })
        return config

# Slovník s cestami k modelům
model_paths = {
    "repeat": "repeat_model.h5",
    "sort":   "sort_model.h5"
}

SEQUENCE_LENGTH = 10  # předpokládaná délka vstupní sekvence

print("Zadejte 'repeat' pro opakování modelu, 'sort' pro třídicí model, nebo 'exit' pro ukončení.\n")

while True:
    model_choice = input("Vyberte model (repeat / sort / exit): ").strip().lower()
    if check_exit(model_choice):
        break

    if model_choice not in model_paths:
        print("Neznámý výběr. Zkuste znovu.\n")
        continue

    model_path = model_paths[model_choice]
    try:
        model = load_model(model_path, custom_objects={
            "TransformerBlock": TransformerBlock,
            "PositionalEncoding": PositionalEncoding
        })
        print(f"Model '{model_choice}' úspěšně načten z: {model_path}")
    except OSError:
        print(f"Chyba: Soubor '{model_path}' se nepodařilo načíst. Zkontrolujte, že existuje.")
        continue

    # Definice předkompilované funkce pro predikci s pevnou signaturou
    # Očekáváme vstup tvaru (None, 10) a typu int32 (protože se jedná o indexy tokenů)
    predict_fn = tf.function(
        lambda x: model(x, training=False),
        input_signature=[tf.TensorSpec(shape=[None, SEQUENCE_LENGTH], dtype=tf.int32)]
    )

    print("\nZadejte 10 čísel (0 - 9) oddělených mezerou, nebo 'exit' pro ukončení.\n")
    input_str = input(">> ")
    if check_exit(input_str):
        break

    try:
        sequence = list(map(int, input_str.strip().split()))
    except ValueError:
        print("Chyba: Zadejte pouze čísla (0-9) oddělená mezerou.\n")
        continue

    if len(sequence) != SEQUENCE_LENGTH:
        print(f"Chyba: Musíte zadat přesně {SEQUENCE_LENGTH} čísel.\n")
        continue

    # Převedeme vstup na konstantní tensor s pevným tvarem a správným dtype
    input_array = tf.constant([sequence], dtype=tf.int32)  # tvar (1, 10)
    predictions = predict_fn(input_array)  # predictions má tvar (1, 10, vocab_size)
    predicted_sequence = tf.argmax(predictions, axis=-1).numpy()[0]

    print("\n--------------------------------------------")
    print("Vstupní sekvence:  ", sequence)
    print("Model predikoval: ", predicted_sequence.tolist())
    print("--------------------------------------------\n")

    # [18, 33, 10, 9, 44, 52, 34, 21, 17, 61]
