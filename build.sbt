name := "PredictStock"

version := "1.0"

scalaVersion := "2.11.8"

val sparkVersion = "2.2.0"

libraryDependencies += "org.apache.spark" %% "spark-core" % sparkVersion
libraryDependencies += "org.apache.spark" %% "spark-mllib" % sparkVersion
libraryDependencies += "edu.stanford.nlp" % "stanford-corenlp" % "latest.integration" artifacts (Artifact("stanford-corenlp", "models"), Artifact("stanford-corenlp"))
libraryDependencies += "org.slf4j" % "slf4j-api" % "latest.integration"
libraryDependencies += "org.slf4j" % "slf4j-simple" % "latest.integration"
libraryDependencies += "org.scalanlp" %% "breeze" % "0.12"
libraryDependencies += "org.scalanlp" %% "breeze-natives" % "0.12"
libraryDependencies += "org.scalanlp" %% "breeze-viz" % "0.12"