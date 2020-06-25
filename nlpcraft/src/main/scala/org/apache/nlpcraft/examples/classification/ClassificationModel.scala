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

package org.apache.nlpcraft.examples.cars

import java.util

import org.apache.nlpcraft.model.{NCIntentTerm, _}

// TODO:
class ClassificationModel extends NCModelFileAdapter("org/apache/nlpcraft/examples/classification/classification_model.yaml") {
    private def mkMap(
        senTotalScore: Double,
        senFtextScore: Double,
        exampleTotalScore: Double,
        exampleFtextScore: Double
    ): java.util.Map[String, Double]  ={
        val m = new java.util.HashMap[String, Double]()

        m.put("min.sentence.total.score", senTotalScore)
        m.put("min.sentence.ftext.score", senFtextScore)
        m.put("min.example.total.score", exampleTotalScore)
        m.put("min.example.ftext.score", exampleFtextScore)

        m
    }

    // Optional.
    override def getMetadata: util.Map[String, AnyRef] = {
        val md = super.getMetadata

        val factors = new java.util.HashMap[String, java.util.Map[String, Double]]()

        factors.put("class:carBrand", mkMap(0.5, 0.5, 0.5, 0.5))
        factors.put("class:animal", mkMap(0.5, 0.5, 0.5, 0.5))
        factors.put("class:weather", mkMap(0.5, 0.5, 0.5, 0.5))

        md.put("ctx.words.factors", factors)

        md
    }

    @NCIntentRef("classification")
    def onMatch(
        @NCIntentTerm("brands") brands: Seq[NCToken],
        @NCIntentTerm("animals") animals: Seq[NCToken],
        @NCIntentTerm("ws") ws: Seq[NCToken]
    ): NCResult = {
        val s =
            if (brands.nonEmpty || animals.nonEmpty || ws.nonEmpty) {
                val s =
                    Map("brands" → brands, "animals" → animals, "weather related" → ws).
                        filter(_._2.nonEmpty).
                        map { case (name, seq) ⇒
                            println(seq)
                            println(seq.map(_.getMetadata))

                            val s = seq.map(p ⇒ s"${p.origText} (${p.metax[Double](s"${p.getId}:score")})").mkString(", ")

                            s"$name=$s"}.mkString(", ")

                s"Classified '$s."
            }
            else
                s"Not found any elements."

        println(s"Result is: $s")

        NCResult.text(s)
    }
}
