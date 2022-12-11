"""The assignment for Data-Intensive Programming 2022"""

from typing import List, Tuple
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from pyspark.ml.feature import VectorAssembler, StringIndexer, MinMaxScaler
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.functions import col, max, sum, lit, when, desc
import matplotlib.pyplot as plt

class Assignment:
    spark: SparkSession = SparkSession.builder \
        .appName("assignment22") \
        .config("spark.driver.host", "localhost") \
        .master("local") \
        .getOrCreate()

    schema1 = StructType([ \
        StructField("a", FloatType(), True),
        StructField("b", FloatType(), True),
        StructField("LABEL", StringType(), True),
        ])
    schema2 = StructType([ \
        StructField("a", FloatType(), True),
        StructField("b", FloatType(), True),
        StructField("c", FloatType(), True),
        StructField("LABEL", StringType(), True),
        ])


    # the data frame to be used in tasks 1 and 4
    dataD2: DataFrame = spark.read.schema(schema1).option("header", "true").csv("data/dataD2.csv").select("a", "b")

    # the data frame to be used in task 2
    dataD3: DataFrame = spark.read.schema(schema2).option("header", "true").csv("data/dataD3.csv")  # REPLACE with actual implementation

    # the data frame to be used in task 3 (based on dataD2 but containing numeric labels)
    dataD2WithLabels: DataFrame = spark.read.schema(schema1).option("header", "true").csv("data/dataD2.csv")

    dataD2_dirty: DataFrame = spark.read.schema(schema1).option("header", "true").csv("data/dataD2_dirty.csv")


    @staticmethod
    def task1(df: DataFrame, k: int) -> List[Tuple[float, float]]:
        """
        this method calculates the mean k error for given dataframe, this method expects the dataframes
        columns to be named as a and b.

        :param df: Dataframe that the is desired for the clusters to be calculated
        :param k: Number of means
        :return: list containing the tuples of cluster centers
        """
        vectorAssembler: VectorAssembler = VectorAssembler(inputCols=["a", "b"], outputCol='features')
        scaler: MinMaxScaler = MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures")
        kmeans_val: KMeans = KMeans(k=k, seed=1, featuresCol="scaledFeatures")
        # Defining the ML pipeline
        pipeline = Pipeline(stages=[vectorAssembler, scaler])
        scaledData = pipeline.fit(df).transform(df)
        centers: list = kmeans_val.fit(scaledData).clusterCenters()
        return list(map(tuple, centers))

    @staticmethod
    def task2(df: DataFrame, k: int) -> List[Tuple[float, float, float]]:
        """
        this method calculates the mean k error for given dataframe 3-dimensional data, this method expects the dataframes
        columns to be named as a, b and c.

        :param df: Dataframe that the is desired for the clusters to be calculated
        :param k: Number of means
        :return: list containing the tuples of cluster centers
        """
        vectorAssembler: VectorAssembler = VectorAssembler(inputCols=["a", "b", "c"], outputCol='features')
        scaler: MinMaxScaler = MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures")
        kmeans_val: KMeans = KMeans(k=k, featuresCol="scaledFeatures", seed=1)
        #Defining the ML pipeline
        pipeline = Pipeline(stages=[vectorAssembler, scaler])
        scaledData = pipeline.fit(df).transform(df)
        centers: list = kmeans_val.fit(scaledData).clusterCenters()

        return list(map(tuple, centers))

    @staticmethod
    def task3(df: DataFrame, k: int) -> List[Tuple[float, float]]:

        # needed modifiers
        string_indexer: StringIndexer = StringIndexer(inputCol='LABEL', outputCol='Label_numeric')
        vectorAssembler: VectorAssembler = VectorAssembler(inputCols=["a", "b", "Label_numeric"], outputCol='features')
        scaler: MinMaxScaler = MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures")
        kmeans_val: KMeans = KMeans(k=k, featuresCol="scaledFeatures", seed=1)
        pipeline = Pipeline(stages=[string_indexer, vectorAssembler, scaler])

        # scaled dataframe
        scaledData: DataFrame = pipeline.fit(df).transform(df)


        model = kmeans_val.fit(scaledData)
        df: DataFrame = model.transform(scaledData) # predicted/clustered dataframe

        most_fatal: DataFrame = df.groupBy("Label_numeric", "prediction").count().where(df.Label_numeric == 1).orderBy(desc(col("count"))).limit(2)
        clusters: List = most_fatal.rdd.map(lambda x: x.prediction).collect()
        centers: list = model.clusterCenters()
        centers: list = [(centers[clusters[0]][0], centers[clusters[0]][1]), (centers[clusters[1]][0], centers[clusters[1]][1])]

        return list(map(tuple, centers))

    # Parameter low is the lowest k and high is the highest one.
    @staticmethod
    def task4(df: DataFrame, low: int, high: int, is_printing=True) -> List[Tuple[int, float]]:
        score = []
        vectorAssembler: VectorAssembler = VectorAssembler(inputCols=["a", "b"], outputCol='features')
        scaler: MinMaxScaler = MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures")
        kmeans_val: KMeans = KMeans(k=low, featuresCol="scaledFeatures", seed=1)
        evaluator: ClusteringEvaluator = ClusteringEvaluator(featuresCol="scaledFeatures").setPredictionCol("prediction").setMetricName("silhouette")

        pipeline = Pipeline(stages=[vectorAssembler, scaler, kmeans_val])

        scaledData: DataFrame = pipeline.fit(df).transform(df)
        silhouetteScore: float = evaluator.evaluate(scaledData)

        score.append((low, silhouetteScore))

        if low == high:
            return score

        else:
            score.extend(Assignment.task4(df, low+1, high, is_printing=False))
            if is_printing:
                x_val = [x[0] for x in score]
                y_val = [x[1] for x in score]
                plt.title("Score as the function of k")
                plt.plot(x_val, y_val, color="red")
                plt.savefig("DIP_plot.png")
            return score

    @staticmethod
    def clean_data_frame(df: DataFrame):
        """
        :param df: dirty frame to be cleaned
        :return: clean frame
        """

        df = df.dropna()
        df = df.filter((df.LABEL != 'Unknown'))

        return df
