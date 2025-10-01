import logging
import os

##loggers several quantities


def logger_f(message,log_path):       
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(log_path)

    if not logger.hasHandlers():  # Prevent duplicate handlers
        logger.setLevel(logging.INFO)

        app_handler = logging.FileHandler(log_path)
        app_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        app_handler.setFormatter(app_formatter)

        logger.addHandler(app_handler)

    logger.info(message)  
    # logger = logging.getLogger(f"{name_file}")
    # logger.setLevel(logging.INFO)

    # app_handler = logging.FileHandler(f"{name_file}.log")
    # app_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # app_handler.setFormatter(app_formatter)
    # logger.addHandler(app_handler)
    # logger.info(message)

# results_coalitions
logger = logging.getLogger("results")
logger.setLevel(logging.INFO)

app_handler = logging.FileHandler("results.log")
app_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
app_handler.setFormatter(app_formatter)
logger.addHandler(app_handler)

#results shapley
svlogger = logging.getLogger("shapley")
svlogger.setLevel(logging.INFO)

svhandler = logging.FileHandler("shapley.log")
svformatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
svhandler.setFormatter(svformatter)
svlogger.addHandler(svhandler)


# results ppce
ppcelogger = logging.getLogger("ppce")
ppcelogger.setLevel(logging.INFO)

ppcehandler = logging.FileHandler("ppce.log")
ppceformatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ppcehandler.setFormatter(ppceformatter)
ppcelogger.addHandler(ppcehandler)


# # results metrics
# metricslogger = logging.getLogger("metrics")
# metricslogger.setLevel(logging.INFO)

# metricshandler = logging.FileHandler("metrics.log")
# metricsformatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# metricshandler.setFormatter(metricsformatter)
# metricslogger.addHandler(metricshandler)

#mean_std metrics
mean_stdlogger = logging.getLogger("mean_std")
mean_stdlogger.setLevel(logging.INFO)

mean_stdhandler = logging.FileHandler("mean_std.log")
mean_stdformatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
mean_stdhandler.setFormatter(mean_stdformatter)
mean_stdlogger.addHandler(mean_stdhandler)



# # Example Usage
# app_logger.info("Application started successfully.")
# user_logger.info("User logged in with ID 12345.")

if __name__=="__main__":
    a=2+1
    message=f"test: {a}"
    name_1="metrics1"
    name_2="metrics2"
    logger_f(message,name_1)
    logger_f(message,name_2)