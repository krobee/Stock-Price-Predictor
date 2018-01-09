package main.scala

import java.util.Properties
import edu.stanford.nlp.ling.CoreAnnotations
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations
import edu.stanford.nlp.pipeline.{ Annotation, StanfordCoreNLP }
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations
import edu.stanford.nlp.process.PTBTokenizer
import edu.stanford.nlp.process.TokenizerFactory

import scala.collection.convert.wrapAll._

// object for sentiment analysis
object SentimentAnalyzer {

  val props = new Properties()
  props.setProperty("annotators", "tokenize, ssplit, parse, sentiment")
  val pipeline: StanfordCoreNLP = new StanfordCoreNLP(props)

  /**
   * Extracts the main sentiment for a given input
   */
  def mainSentiment(input: String): Int = Option(input) match {
    case Some(text) if text.nonEmpty => extractSentiment(text)
    case _                           => throw new IllegalArgumentException("input can't be null or empty")
  }

  /**
   * Extracts a list of sentiments for a given input
   */
  def sentiment(input: String): List[(String, Int)] = Option(input) match {
    case Some(text) if text.nonEmpty => extractSentiments(text)
    case _                           => throw new IllegalArgumentException("input can't be null or empty")
  }

  private def extractSentiment(text: String): Int = {
    val (_, sentiment) = extractSentiments(text)
      .maxBy { case (sentence, _) => sentence.length }
    sentiment
  }

  private def extractSentiments(text: String): List[(String, Int)] = {
    val annotation: Annotation = pipeline.process(text)
    val sentences = annotation.get(classOf[CoreAnnotations.SentencesAnnotation])
    sentences
      .map(sentence => (sentence, sentence.get(classOf[SentimentCoreAnnotations.SentimentAnnotatedTree])))
      .map { case (sentence, tree) => (sentence.toString, RNNCoreAnnotations.getPredictedClass(tree)) }
      .toList
  }

}