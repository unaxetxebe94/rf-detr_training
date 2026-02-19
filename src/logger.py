import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

def get_logger(
    name: str = "app",
    log_file: str = "logs/app.log",
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    console: bool = True,
    fmt: Optional[str] = None,
) -> logging.Logger:
    """
    Crea/retorna un logger configurado que escribe a archivo (rotating) y opcionalmente a consola.

    Parámetros:
    - name: nombre del logger (usa normalmente __name__ en módulos).
    - log_file: ruta al archivo de log.
    - level: nivel de logging (logging.INFO, DEBUG, WARNING, ...).
    - max_bytes: tamaño máximo del archivo antes de rotar.
    - backup_count: cuántos archivos rotados conservar.
    - console: si True, añade StreamHandler para ver logs por consola.
    - fmt: formato de mensaje (si None se usa el formato por defecto).
    """
    if fmt is None:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"

    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    logger = logging.getLogger(name)

    # Evitar reconfigurar si ya está hecho
    if getattr(logger, "_configured", False):
        return logger

    logger.setLevel(level)
    logger.propagate = False  # evita doble logging si root tiene handlers

    # Asegurar que el directorio del log exista
    log_path = Path(log_file)
    if log_path.parent:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # File handler (rotating)
    file_handler = RotatingFileHandler(
        filename=str(log_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    file_handler.name = "file"
    logger.addHandler(file_handler)

    # Console handler (opcional)
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        console_handler.name = "console"
        logger.addHandler(console_handler)

    # Marca como configurado para evitar añadir handlers varias veces
    logger._configured = True
    return logger
