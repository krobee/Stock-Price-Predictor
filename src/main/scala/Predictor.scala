package main.scala

import java.io._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.{ IntegerType, FloatType}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegressionModel

import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.fs.Path

object Predictor {

  def main(args: Array[String]) {
    // local mode for testing
    val spark = SparkSession.builder
      .master("local")
      .appName("TweetSentiment")
      .getOrCreate

    // cluster mode for deploying
    //    val spark = SparkSession.builder
    //      .master("spark://jupiter:31106")
    //      .appName("TweetSentiment")
    //      .getOrCreate

    // read arguments
    val code = args(0)
    val startDate = args(1).toInt
    val endDate = args(2).toInt

    var dfSentiment = spark.read
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("mode", "DROPMALFORMED")
      .load("hdfs://jupiter:31101/cs535_data/SentimentCSV/sentiment.csv");
    
    
    var daysDelay = 1
    dfSentiment = dfSentiment.where(col("Code") === code && col("Date").between(startDate - daysDelay, endDate - daysDelay))
       .withColumn("Total", col("Total").cast(IntegerType))
      .withColumn("AvgValue", col("AvgValue").cast(FloatType))
    
    // read stockprice.csv
    var dfStockprice = spark.read
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("mode", "DROPMALFORMED")
      .load("hdfs://jupiter:31101/cs535_data/stockprice.csv");

    // convert column type
    dfStockprice = dfStockprice.withColumn("Close", col("Close").cast(FloatType))
      .withColumn("Offset", col("Offset").cast(FloatType)+1)
      .withColumn("Date", col("Date").cast(IntegerType))
      .withColumnRenamed("Date", "Stock Date")
      .withColumnRenamed("Code", "Stock Code")
      
      
    
    
    dfStockprice = dfStockprice.join(dfSentiment, col("Date") + daysDelay === col("Stock Date") && col("Code") === col("Stock Code"))
      .where(dfStockprice("Offset").isNotNull)
      
      
     // convert to dense vector feature
    val assembler = new VectorAssembler()
      .setInputCols(Array("Close", "AvgValue", "Total"))
      .setOutputCol("features") 
      
    // drop unnecessary columns
    // finish data pre-processing
    val data = assembler.transform(dfStockprice)
      .drop(col("Stock Code"))
      .withColumnRenamed("Offset", "label")
      
    data.show()
    
    val dirModel = "hdfs://jupiter:31101/cs535_data/Model"
    val lrModel = LinearRegressionModel.load(dirModel)
    
    var dfPredicted = lrModel.transform(data)
   
     
      
    dfPredicted = dfPredicted.withColumn("Predicted Price", col("Close")/col("label")*col("prediction"))
      .withColumn("Diff", col("Predicted Price")-col("Close"))
      .select("Stock Date", "Code", "Close", "Predicted Price", "Diff")
      .orderBy(asc("Stock Date"))
//    predicted.show()
    
    val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)
    val dirPredict = "hdfs://jupiter:31101/cs535_data/Prediction"
    
    if(fs.exists(new Path(dirPredict)))
       fs.delete(new Path(dirPredict),true)
    
    
    dfPredicted.coalesce(1)
      .write
      .option("header", "true")
      .csv(dirPredict)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
  }
}