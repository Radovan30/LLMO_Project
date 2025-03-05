# Transformer-Based Sequence Processing in TensorFlow

Tento projekt implementuje **Transformer modely** v TensorFlow pro **učení sekvencí**, konkrétně pro dvě úlohy:  
1. **Opakování sekvence** (`repeat`) – model se učí vrátit vstupní sekvenci beze změny.  
2. **Řazení sekvence** (`sort`) – model se učí seřadit vstupní sekvenci vzestupně.

## Funkce projektu
- **Vlastní Transformer vrstvy**:
  - `PositionalEncoding`: Implementuje poziční kódování pro zachování pořadí sekvence.
  - `TransformerBlock`: Multi-head attention + feedforward síť s normalizací.
- **Interaktivní uživatelské rozhraní** v konzoli:
  - Funguje v rozmezí čísel (od `0` do `100`).  
  - Výběr modelu (`repeat` nebo `sort`).
  - Zadání vstupní sekvence (`10 čísel` oddělených mezerou).
  - Model provede predikci a zobrazí výsledek.
- **Načítání předtrénovaných modelů** (`repeat_model.h5` a `sort_model.h5`).
- **Použití TensorFlow pro optimalizovanou inference** pomocí `tf.function`.


## anaconda packages in environment 
pip install -r requirements.txt
