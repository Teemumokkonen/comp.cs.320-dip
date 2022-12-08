"""The assignment for Data-Intensive Programming 2022"""

from typing import List, Tuple
from pyspark.ml.clustering import KMeans
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, MinMaxScaler
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.functions import col, max, sum, lit, when, desc

class Assignment:
    spark: SparkSession = SparkSession.builder \
        .appName("assignment22") \
        .config("spark.driver.host", "localhost") \
        .master("local") \
        .getOrCreate()

    # the data frame to be used in tasks 1 and 4
    dataD2: DataFrame = spark.read.option("inferSchema", "true").option("header", "true").csv("data/dataD2.csv").select("a", "b")

    # the data frame to be used in task 2
    dataD3: DataFrame = spark.read.option("inferSchema", "true").option("header", "true").csv("data/dataD3.csv")  # REPLACE with actual implementation

    # the data frame to be used in task 3 (based on dataD2 but containing numeric labels)
    dataD2WithLabels: DataFrame = spark.read.option("inferSchema", "true").option("header", "true").csv("data/dataD2.csv")


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
        kmeans_val = KMeans(k=k, seed=1, featuresCol="scaledFeatures")
        assembled_df = vectorAssembler.transform(df)
        scaler = MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures")
        scalerModel = scaler.fit(assembled_df)
        scaledData = scalerModel.transform(assembled_df)
        model = kmeans_val.fit(scaledData)
        centers = model.clusterCenters()
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
        assembled_df = vectorAssembler.transform(df)

        # scale the data

        scaler = MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures")
        scalerModel = scaler.fit(assembled_df)
        scaledData = scalerModel.transform(assembled_df)

        kmeans_val = KMeans(k=k, featuresCol="scaledFeatures", seed=1)
        model = kmeans_val.fit(scaledData)
        centers = model.clusterCenters()
        return list(map(tuple, centers))

    @staticmethod
    def task3(df: DataFrame, k: int) -> List[Tuple[float, float]]:
        string_indexer = StringIndexer(inputCol='LABEL', outputCol='Label_numeric')
        df = string_indexer.fit(df).transform(df)
        vectorAssembler: VectorAssembler = VectorAssembler(inputCols=["a", "b", "Label_numeric"], outputCol='features')
        kmeans_val = KMeans(k=k, featuresCol="scaledFeatures", seed=1)
        dataD2WithLabels = vectorAssembler.transform(df)
        scaler = MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures")
        scalerModel = scaler.fit(dataD2WithLabels)
        scaledData = scalerModel.transform(dataD2WithLabels)
        model = kmeans_val.fit(scaledData)
        df = model.transform(scaledData) # predicted/clustered dataframe
        most_fatal = df.groupBy("Label_numeric", "prediction").count().where(df.Label_numeric == 1).orderBy(desc(col("count"))).limit(2)
        clusters = most_fatal.rdd.map(lambda x: x.prediction).collect()
        centers = model.clusterCenters()
        centers = [(centers[clusters[0]][0], centers[clusters[0]][1]), (centers[clusters[1]][0], centers[clusters[1]][1])]


        return list(map(tuple, centers))

    # Parameter low is the lowest k and high is the highest one.
    @staticmethod
    def task4(df: DataFrame, low: int, high: int) -> List[Tuple[int, float]]:
        evaluator = ClusteringEvaluator(featuresCol="scaledFeatures").setPredictionCol("prediction").setMetricName("silhouette")
        vectorAssembler: VectorAssembler = VectorAssembler(inputCols=["a", "b"], outputCol='features')
        assembled_df = vectorAssembler.transform(df)
        scaler = MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures")
        scalerModel = scaler.fit(assembled_df)
        scaledData = scalerModel.transform(assembled_df)
        scaledData.show()
        score = []

        for i in range(low, high + 1):
            kmeans_val = KMeans(k=i, featuresCol="scaledFeatures", seed=1)
            model = kmeans_val.fit(scaledData)
            predictions = model.transform(scaledData)
            silhouetteScore = evaluator.evaluate(predictions)
            score.append((i, silhouetteScore))
        return score