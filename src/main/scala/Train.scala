package main.scala

import java.io._
import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.fs.Path

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{ IntegerType, FloatType }

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression

object Train {
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
      
    var dfSentiment = spark.read
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("mode", "DROPMALFORMED")
      .load("hdfs://jupiter:31101/cs535_data/SentimentCSV/sentiment.csv");
    
    dfSentiment = dfSentiment.withColumn("Date", col("Date").cast(IntegerType))
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

    // merge tweets and stockprice
      
    // 1 2 0 3
    var daysDelay = 1 // set later
    
    dfStockprice = dfStockprice.join(dfSentiment, col("Date") + daysDelay === col("Stock Date") && col("Code") === col("Stock Code"))
      .where(dfStockprice("Offset").isNotNull)
      
   // convert to dense vector feature
    val assembler = new VectorAssembler()
      .setInputCols(Array("Close", "AvgValue", "Total"))
      .setOutputCol("features")

    // drop unnecessary columns
    // finish data pre-processing
    val data = assembler.transform(dfStockprice)
      .drop("Stock Code")
      .withColumnRenamed("Offset", "label")

    // split training and test dataset
    val splits = data.randomSplit(Array(0.8, 0.2))
    val training = splits(0).cache()
    val test = splits(1).cache()

    val numTraining = training.count()
    val numTest = test.count()

    // training
    val lr = new LinearRegression()
    val lrModel = lr.fit(training)
    
    // testing
    val predicted = lrModel.transform(test)

    // summarize the model
    val trainingSummary = lrModel.summary
    
    val pw = new PrintWriter(new File("TestResult.txt"))
    pw.write("NumTraining: " + numTraining + " NumTest: " + numTest + "\n")
    pw.write(trainingSummary.rootMeanSquaredError.toString() + "\n")
    pw.write(trainingSummary.r2.toString())
    pw.close
    
    // save test result
    val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)
    val dirTest = "hdfs://jupiter:31101/cs535_data/TestPrediction"
    
    if(fs.exists(new Path(dirTest)))
       fs.delete(new Path(dirTest),true)
       
    predicted.rdd.coalesce(1).saveAsTextFile(dirTest)
    
    // save model
    val dirModel = "hdfs://jupiter:31101/cs535_data/Model"
    if(fs.exists(new Path(dirModel)))
       fs.delete(new Path(dirModel),true)
    
    lrModel.save(dirModel)
  }
}