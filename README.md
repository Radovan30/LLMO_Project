# Transformer-Based Sequence Processing in TensorFlow

Tento projekt implementuje **Transformer modely** v TensorFlow pro **učení sekvencí**, konkrétně pro dvě úlohy:  
1. **Opakování sekvence** (`repeat`) – model se učí vrátit vstupní sekvenci beze změny.  
2. **Řazení sekvence** (`sort`) – model se učí seřadit vstupní sekvenci vzestupně.

## Funkce projektu
- **Vlastní Transformer vrstvy**:
  - `PositionalEncoding`: Implementuje poziční kódování pro zachování pořadí sekvence.
  - `TransformerBlock`: Multi-head attention + feedforward síť s normalizací.
- **Interaktivní uživatelské rozhraní** v konzoli:
  - Výběr modelu (`repeat` nebo `sort`).
  - Zadání vstupní sekvence (`10 čísel` oddělených mezerou).
  - Model provede predikci a zobrazí výsledek.
- **Načítání předtrénovaných modelů** (`repeat_model.h5` a `sort_model.h5`).
- **Použití TensorFlow pro optimalizovanou inference** pomocí `tf.function`.


# anaconda packages in environment 
#
# Name                    Version                   
jpeg                      9e                     
keras                     2.10.0                  
keras-preprocessing       1.1.2                                 
libbrotlicommon           1.0.9                  
libbrotlidec              1.0.9                  
libbrotlienc              1.0.9                       
matplotlib                3.9.2              
matplotlib-base           3.9.2              
mkl                       2023.1.0           
mkl-service               2.4.0              
mkl_fft                   1.3.10             
mkl_random                1.2.7              
numpy                     1.26.4             
numpy-base                1.26.4                          
openjpeg                  2.5.2                  
openssl                   3.3.2               
opt-einsum                3.4.0                   
packaging                 24.1               
pillow                    10.4.0             
pip                       24.3.1                  
ply                       3.11               
protobuf                  3.19.6                   
pyasn1                    0.6.1                   
pyasn1-modules            0.4.1                        
pybind11-abi              5                      
pyparsing                 3.1.2              
pyqt                      5.15.10            
pyqt5-sip                 12.13.0            
python                    3.9.20                 
python-dateutil           2.9.0post0                  
tensorboard               2.10.1                  
tensorboard-data-server   0.6.1                                   
tensorflow                2.10.0                   
tensorflow-estimator      2.10.0                  
tensorflow-io-gcs-filesystem 0.31.0                
             
