import os
import sys

# Cấu hình Python Worker trên Windows
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Cấu hình kết nối local
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("THPT_BigData_2026")
    .master("local[2]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.python.worker.reuse", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# Kiểm tra Python Worker
rdd = spark.sparkContext.parallelize([1, 2, 3, 4, 5], 2)

result = rdd.map(lambda x: x * 2).collect()

print("Ket qua Python Worker:", result)

spark.stop()