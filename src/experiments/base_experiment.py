import os
import abc
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import hydra

from src.utils import LoggerMetrics, dataclass_json_dump
from src.visualization import StandardRenderer
from src.time_tracker import TimeTracker

# Configurar logger global
log = logging.getLogger(__name__)

class BaseExperiment(abc.ABC):
    """
    Clase base abstracta para todos los experimentos.
    Maneja la configuración, logging, métricas y ciclo de vida del experimento.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.output_dir = Path(os.getcwd())
        
        # Inicializar utilidades
        self.metrics = LoggerMetrics()
        self.time_tracker = TimeTracker()
        self.renderer = None
        
        # Estado del experimento
        self.env = None
        self.robot = None
        self.controller = None
        self.finished = False
        
        log.info(f"Inicializando experimento: {cfg.experiment.name}")
        log.info(f"Directorio de salida: {self.output_dir}")
        
        # Guardar configuración completa
        with open("config_dump.yaml", "w") as f:
            OmegaConf.save(cfg, f)

    @abc.abstractmethod
    def setup(self):
        """
        Configura el entorno, robot y controladores.
        Debe ser implementado por las subclases.
        """
        pass

    @abc.abstractmethod
    def run_episode(self) -> Dict[str, Any]:
        """
        Ejecuta un episodio completo de simulación.
        Debe ser implementado por las subclases.
        """
        pass

    def setup_renderer(self, opts: Dict[str, Any]):
        """Inicializa el renderizador estándar."""
        if self.cfg.env.render:
            self.renderer = StandardRenderer(opts=opts)

    def teardown(self):
        """Limpieza de recursos y guardado de resultados finales."""
        log.info("Finalizando experimento y guardando resultados...")
        
        # Guardar métricas
        metrics_path = self.output_dir / "metrics.csv"
        self.metrics.dump_to_csv(str(metrics_path))
        log.info(f"Métricas guardadas en {metrics_path}")
        
        # Guardar tiempos
        time_path = self.output_dir / "timing.csv"
        time_df = pd.DataFrame(self.time_tracker.as_dict())
        time_df.to_csv(time_path, index=False)
        
        # Cerrar entorno y renderer
        if self.env:
            self.env.close()
        
        if self.renderer:
            self.renderer.close() # Asegura que se guarden los videos

    def run(self):
        """Método principal para ejecutar el experimento."""
        try:
            self.setup()
            results = self.run_episode()
            return results
        except Exception as e:
            log.error(f"Error durante la ejecución del experimento: {e}", exc_info=True)
            raise
        finally:
            self.teardown()

    # --- Métodos Helpers Comunes ---

    def get_initial_location(self) -> np.ndarray:
        """Selecciona una posición inicial aleatoria de la configuración."""
        locs = self.cfg.experiment.initial_locations
        idx = np.random.randint(0, len(locs))
        return np.array(locs[idx])

    def check_termination(self, state, reward, terminated, truncated) -> bool:
        """Verifica condiciones de terminación y loguea el estado final."""
        if terminated or truncated:
            status = 'truncated'
            if terminated:
                status = 'reached_goal' if reward > 0.0 else 'crashed'
            
            self.metrics.add_value('status', status)
            log.info(f"Episodio terminado. Estado: {status}")
            return True
        
        self.metrics.add_value('status', 'running')
        return False

    def log_common_metrics(self, state, action, t):
        """Registra métricas estándar comunes a la mayoría de experimentos."""
        # Distancia al objetivo
        if hasattr(self.env, 'env_params') and self.env.env_params.goal_location is not None:
            dist = np.linalg.norm(state["location"] - self.env.env_params.goal_location)
            self.metrics.add_value('dist_to_goal', dist)

        # Humo en el robot
        if hasattr(self.env, 'get_smoke_density_in_robot'):
            smoke = self.env.get_smoke_density_in_robot()[0]
            self.metrics.add_value('smoke_on_robot', smoke)
            
            # Acumulado
            last_acc = self.metrics.get_last_value('smoke_on_robot_acc')
            self.metrics.add_value('smoke_on_robot_acc', last_acc + smoke)

        # Acciones y Estado
        for i, a in enumerate(action):
            self.metrics.add_value(f'action_{i}', a)
        
        for i, s in enumerate(state["location"]):
            self.metrics.add_value(f'state_{i}', s)
            
        self.metrics.add_value('steps', t)
        self.metrics.add_value('time', t * self.cfg.env.clock)
