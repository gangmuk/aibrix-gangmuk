import logging
import os

# Configure logging
def setup_logging():
    logging_level = os.environ.get("LOG_LEVEL", "INFO")
    
    # Add %(filename)s to the format string to include the file name
    # print function name as well
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("llm_router.log")
        ]
    )

    # logging.basicConfig(level=getattr(logging, logging_level), format='%(asctime)s - %(levelname)s - %(filename)s - %(message)s')

    
    
    logger = logging.getLogger("llm_router")
    return logger

# Create and export a common logger instance
logger = setup_logging()