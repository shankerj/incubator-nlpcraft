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

import org.apache.nlpcraft.model.{NCIntentTerm, _}

// TODO:
class ClassificationModel extends NCModelFileAdapter("org/apache/nlpcraft/examples/classification/classification_model.yaml") {
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
