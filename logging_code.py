import logging
import sys

def setup_logging(script_name):
    try:
        # Logger object banana script name ke saath
        logger = logging.getLogger(script_name)

        if not logger.handlers:
            # Log level set karna (DEBUG mode)
            logger.setLevel(logging.DEBUG)

            # Log file ka path
            log_path = f'C:\\Users\\Nazneen\\OneDrive\\Desktop\\AI-Powered Customer Retention Prediction System\\logs\\{script_name}.log'

            # File handler banana (UTF-8 encoding ke saath taake special symbols error na dein)
            handler = logging.FileHandler(log_path, mode='w', encoding='utf-8') # <--- Yeh change zaroori hai

            # Log ka format set karna
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)

            # Logger mein handler add karna
            logger.addHandler(handler)

        # Logger ko propagate hone se rokna taake duplicate logs na aayein [1]
        logger.propagate = False
        return logger

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        print(f"Error in logging setup at line {er_line.tb_lineno}: {er_msg}")