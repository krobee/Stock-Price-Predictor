package main.scala

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{ IntegerType, FloatType }
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.udf

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.{ LongWritable, Text }
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat
import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.fs.Path


object TweetSentiment {
  def main(args: Array[String]) {
     // local mode for testing
//        val spark = SparkSession.builder
//          .master("local")
//          .appName("TweetSentiment")
//          .getOrCreate

    // cluster mode for deploying
    val spark = SparkSession.builder
      .master("spark://jupiter:31106")
      .appName("TweetSentiment")
      .getOrCreate

    import spark.implicits._

    // read trademark.csv
    val dfTrademark = spark.read
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("mode", "DROPMALFORMED")
      .load("hdfs://jupiter:31101/cs535_data/trademarks.csv");

    // read tweets.csv
    // set delimiter
    spark.sparkContext.hadoopConfiguration.set("textinputformat.record.delimiter", "\n\n")
//    val rawTweets = spark.sparkContext.newAPIHadoopFile("hdfs://jupiter:31101/cs535_data/tweets2009-12.txt.gz", classOf[TextInputFormat], classOf[LongWritable], classOf[Text], spark.sparkContext.hadoopConfiguration)
    val rawTweets = spark.sparkContext.newAPIHadoopFile("hdfs://jupiter:31101/cs535_data/twitter500.txt", classOf[TextInputFormat], classOf[LongWritable], classOf[Text], spark.sparkContext.hadoopConfiguration)

    // convert to dataframe <date, username, tweet>
    val tweets = rawTweets.filter(s => (s._2.toString().split("\n").size == 3))
      .map(x => (x._2.toString().split("\n")(0).split("\t")(1).split(" ")(0).replaceAll("-", ""),
        x._2.toString().split("\n")(2).split("\t")(1).trim()))

    var dfTweets = spark.createDataFrame(tweets)

    // convert and rename column
    dfTweets = dfTweets.withColumnRenamed("_1", "Date")
      .withColumnRenamed("_2", "Tweet")

    // merge trademark and tweets
    dfTweets = dfTweets.crossJoin(dfTrademark).filter(col("Tweet").contains(col("trademark")))

    // get sentiment value
    val getSentimentValue = udf { s: String => SentimentAnalyzer.mainSentiment(s) }
    dfTweets = dfTweets.withColumn("Value", getSentimentValue('Tweet).cast(FloatType))
      .drop("Tweet")
      .drop("trademark")

    // group by Date, Code
    // AvgValue = sum(Value)/Total
    dfTweets = dfTweets.groupBy("Date", "Code")
      .agg(sum(col("Value")).alias("SumValue"), count(lit(1)).alias("Total"))
      .withColumn("AvgValue", col("SumValue") / col("Total"))
      .drop("SumValue")

    // save as csv file
    val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)
    val dirSentiment = "hdfs://jupiter:31101/cs535_data/Sentiment"
    
    if(fs.exists(new Path(dirSentiment)))
       fs.delete(new Path(dirSentiment),true)
       
    dfTweets.coalesce(1)
      .write
      .option("header", "true")
      .csv("hdfs://jupiter:31101/cs535_data/Sentiment")
  }
}