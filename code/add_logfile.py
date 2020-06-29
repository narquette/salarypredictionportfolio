import datetime 
import logging
        
now = datetime.datetime.now()
file_name = f"./logs/RandomForest_FinalModel_{now.year}_{now.month}_{now.day}.log"
logging.basicConfig(format='%(asctime)s %(message)s', filename=file_name, level=logging.DEBUG)