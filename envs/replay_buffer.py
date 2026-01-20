import numpy as np
from collections import deque
from typing import Dict, Any, List

class GenericReplayBuffer:
    """
    Un buffer de reproducción genérico que almacena muestras como un diccionario de deques.
    Permite manejar un número variable de tipos de datos (keys).
    """
    
    def __init__(self, buffer_size: int, data_keys: List[str]):
        """
        Inicializa el buffer.
        
        :param buffer_size: El tamaño máximo del buffer.
        :param data_keys: Una lista de strings con las claves de los datos a almacenar.
        """
        self.buffer_size = int(buffer_size)
        self.data_keys = data_keys
        self.buffer: Dict[str, deque] = {}
        self.current_size = 0
        self.reset()

    def reset(self):
        """Inicializa todos los deques para las claves especificadas."""
        self.buffer = {key: deque(maxlen=self.buffer_size) for key in self.data_keys}
        self.current_size = 0

    def add(self, **kwargs: Any):
        """
        Agrega una muestra al buffer usando argumentos de palabras clave.
        Ejemplo: buffer.add(state=s, action=a, done=d)
        
        :param kwargs: Los pares clave-valor que se añadirán al buffer.
                       Las claves deben coincidir con self.data_keys.
        """
        # Asegúrate de que solo se pasen las claves definidas
        if set(kwargs.keys()) != set(self.data_keys):
            raise ValueError(f"Las claves proporcionadas no coinciden con las claves definidas del buffer: {self.data_keys}")
            
        for key, value in kwargs.items():
            self.buffer[key].append(value)
            
        # Actualiza el tamaño actual (solo necesario una vez por adición)
        self.current_size = len(self.buffer[self.data_keys[0]])
        
    def get_from_index(self, index: int) -> Dict[str, Any]:
        """
        Recupera una muestra completa de un índice específico.
        """
        if index >= self.current_size:
            raise IndexError("Índice fuera de los límites del buffer actual.")

        data = {key: self.buffer[key][index] for key in self.data_keys}
        return data

    def save_to_file(self, filepath: str):
        """
        Guarda el contenido del buffer en un archivo .npz.
        """
        data_to_save = {}
        for key in self.data_keys:
            # Convierte deque a un array de numpy
            # Usa dtype=object si los elementos son de tamaño o tipo variable (como arrays/listas)
            # o si se requiere para compatibilidad con la carga.
            temp_array = np.array(list(self.buffer[key]), dtype=object) 
            data_to_save[key] = temp_array

        np.savez(filepath, **data_to_save, buffer_size=self.buffer_size, data_keys=self.data_keys)
        print(f"Buffer guardado exitosamente en: {filepath}")

    def load_from_file(self, filepath: str):
        """
        Carga el contenido del buffer desde un archivo .npz.
        """
        npzfile = np.load(filepath, allow_pickle=True) # allow_pickle es clave para dtype=object
        
        # 1. Cargar metadatos
        if 'data_keys' in npzfile and 'buffer_size' in npzfile:
            loaded_keys = list(npzfile['data_keys'])
            loaded_buffer_size = npzfile['buffer_size'].item()
        else:
            print("El archivo no contiene las claves 'data_keys' y 'buffer_size'.")
            loaded_keys = list(npzfile.keys())
            loaded_buffer_size = len(npzfile[loaded_keys[0]])
        
        if loaded_keys != self.data_keys or loaded_buffer_size != self.buffer_size:
             print(f"ADVERTENCIA: Las claves cargadas ({loaded_keys}) o el tamaño ({loaded_buffer_size}) \n"
                   f"no coinciden con la configuración actual ({self.data_keys}, {self.buffer_size}). \n"
                   f"Usando la configuración y datos cargados.")
             self.data_keys = loaded_keys
             self.buffer_size = loaded_buffer_size

        # 2. Reconstruir los deques
        self.reset()
        for key in self.data_keys:
            if key in npzfile:
                # El numpy array se convierte a lista y luego se carga en el deque
                loaded_list = npzfile[key].tolist() 
                self.buffer[key] = deque(loaded_list, maxlen=self.buffer_size)
            else:
                print(f"ADVERTENCIA: La clave '{key}' no se encontró en el archivo NPZ.")

        self.current_size = len(self.buffer[self.data_keys[0]])
        print(f"Buffer cargado exitosamente. Tamaño actual: {self.current_size}")


    def full(self) -> bool:
        """Verifica si el buffer está lleno."""
        return self.current_size >= self.buffer_size
    
    def __len__(self) -> int:
        """Devuelve el tamaño actual del buffer."""
        return self.current_size

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Permite acceder al buffer como buffer[index]."""
        return self.get_from_index(index)

# # --- EJEMPLO DE USO ---
# print("--- Inicialización y Adición de Muestras ---")
# # 1. Definir las claves que queremos almacenar (similar a tu ejemplo)
# keys_to_store = ['state', 'action', 'smoke_values', 'smoke_value_positions', 'done']

# # 2. Crear una instancia del buffer genérico
# buffer_size = 1000
# replay_buffer = GenericReplayBuffer(buffer_size=buffer_size, data_keys=keys_to_store)

# # 3. Datos de ejemplo con diferentes formas (como podrías tener en tu caso)
# s1 = np.array([1.0, 2.0, 3.0])
# v1 = [10.1, 20.2]
# p1 = (1, 5)

# s2 = np.array([4.0, 5.0, 6.0])
# v2 = [30.3, 40.4, 50.5]
# p2 = (8, 12, 20)

# # 4. Añadir las muestras usando las claves definidas
# replay_buffer.add(
#     state=s1, 
#     action=0, 
#     smoke_values=v1, 
#     smoke_value_positions=p1, 
#     done=False
# )

# replay_buffer.add(
#     state=s2, 
#     action=1, 
#     smoke_values=v2, 
#     smoke_value_positions=p2, 
#     done=True
# )

# print(f"Tamaño actual del buffer: {len(replay_buffer)}")
# print("\n--- Recuperar una Muestra ---")
# sample_0 = replay_buffer[0]
# print(f"Muestra en índice 0:\n{sample_0}")

# # --- Guardar y Cargar ---
# print("\n--- Guardar y Cargar ---")
# filepath = 'generic_replay_data.npz'
# replay_buffer.save_to_file(filepath)

# # Crear un nuevo buffer para cargar los datos
# new_buffer = GenericReplayBuffer(buffer_size=500, data_keys=keys_to_store) # Notar el buffer_size diferente
# new_buffer.load_from_file(filepath)

# print(f"Tamaño actual del nuevo buffer después de cargar: {len(new_buffer)}")
# print(f"Muestra 1 cargada:\n{new_buffer[1]}")

# import numpy as np
# from collections import deque

# # Collect samples and store in numpy. s, a, s', done
# class SmokeReplayBuffer:
#     def __init__(self, buffer_size):
#         self.buffer_size = int(buffer_size)
#         self.reset()

#     def reset(self):
#         self.actions = deque(maxlen=self.buffer_size)
#         self.state = deque(maxlen=self.buffer_size)
#         self.smoke_values = deque(maxlen=self.buffer_size)
#         self.smoke_value_positions = deque(maxlen=self.buffer_size)
#         self.done = deque(maxlen=self.buffer_size)

#     def load_from_file(self, filepath: str):
#         npzfile = np.load(filepath)
#         self.actions = npzfile['actions']
#         self.state = npzfile['state']
#         self.smoke_values = npzfile['smoke_values']
#         self.smoke_value_positions = npzfile['smoke_value_positions']
#         self.done = npzfile['done']

#     def save_to_file(self, filepath: str):
#         data = {
#             'actions': np.array(self.actions),
#             'state': np.array(self.state, dtype=object), # <--- CAMBIO CLAVE
#             'smoke_values': np.array(self.smoke_values, dtype=object), # <--- CAMBIO CLAVE
#             'smoke_value_positions': np.array(self.smoke_value_positions, dtype=object), # Asumiendo que también podría ser variable
#             'done': np.array(self.done)
#         }
#         np.savez(filepath, **data)

#     def add(self, state, action, smoke_values, smoke_value_positions, done):
#         self.actions.append(action)
#         self.state.append(state)
#         self.smoke_values.append(smoke_values)
#         self.smoke_value_positions.append(smoke_value_positions)
#         self.done.append(done)

#     def get_from_index(self, index):
#         data = {
#             "action": self.actions[index], 
#             "state": self.state[index], 
#             "smoke_values": self.smoke_values[index], 
#             "smoke_value_positions": self.smoke_value_positions[index], 
#             "done": self.done[index]
#         }
#         return data

#     def full(self):
#         return len(self.actions) >= self.buffer_size
