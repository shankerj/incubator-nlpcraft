/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.nlpcraft.server.nlp.enrichers.ctxword

import java.{lang, util}

import com.typesafe.scalalogging.LazyLogging

import scala.collection.JavaConverters._
import scala.collection._

object NCContextWordFactors {
    // Configuration when we try to find context words for words nouns using initial sentence.
    private final val MIN_SENTENCE_SCORE = 0.5
    private final val MIN_SENTENCE_FTEXT = 0.5

    // Configuration when we try to find context words for words nouns using substituted examples.
    private final val MIN_EXAMPLE_SCORE = 0.8
    private final val MIN_EXAMPLE_FTEXT = 0.5
}

import NCContextWordFactors._

case class NCContextWordFactors(meta: Map[String, AnyRef], elementIds: Set[String]) extends LazyLogging {
    private val (totalSen, totalExample, ftextSen, ftextExample) = initialize

    private val minTotalSen: Double = minValue(totalSen)
    private val minTotalExample: Double = minValue(totalExample)
    private val minFtextSen: Double = minValue(ftextSen)
    private val minFtextExample: Double = minValue(ftextExample)

    private def initialize: (Map[String, Double], Map[String, Double], Map[String, Double], Map[String, Double]) = {
        def mkMap(d: Double): mutable.Map[String, Double] =
            mutable.HashMap.empty[String, Double] ++ elementIds.map(id ⇒ id → d).toMap

        val totalSen = mkMap(MIN_SENTENCE_SCORE)
        val totalExample = mkMap(MIN_EXAMPLE_SCORE)
        val ftextSen = mkMap(MIN_SENTENCE_FTEXT)
        val ftextExample = mkMap(MIN_EXAMPLE_FTEXT)

        meta.get("ctx.words.factors") match {
            case Some(v) ⇒
                v.asInstanceOf[util.HashMap[String, util.Map[String, lang.Double]]].asScala.
                    foreach { case (elemId, factors) ⇒
                        if (elementIds.contains(elemId)) {
                            def set(name: String, m: mutable.Map[String, Double]): Unit = {
                                val v = factors.get(name)

                                if (v != null)
                                    m += elemId → v
                            }

                            set("min.sentence.total.score", totalSen)
                            set("min.sentence.ftext.score", totalExample)
                            set("min.example.total.score", ftextSen)
                            set("min.example.ftext.score", ftextExample)
                        }
                        else
                            logger.warn(s"Unexpected element ID: '$elemId' data skipped.")
                    }

            case None ⇒ // No-op.
        }

        (totalSen.toMap, totalExample.toMap, ftextSen.toMap, ftextExample.toMap)
    }

    private def minValue(m: Map[String, Double]): Double = m.values.min

    def getMinTotalSentence: Double = minTotalSen
    def getMinTotalExample: Double = minTotalExample
    def getMinFtextSentence: Double = minFtextSen
    def getMinFtextExample: Double = minFtextExample

    def getTotalSentence(elemId: String): Double = totalSen(elemId)
    def getTotalExample(elemId: String): Double = totalExample(elemId)
    def getFtextSentence(elemId: String): Double = ftextSen(elemId)
    def getFtextExample(elemId: String): Double = ftextExample(elemId)
}
