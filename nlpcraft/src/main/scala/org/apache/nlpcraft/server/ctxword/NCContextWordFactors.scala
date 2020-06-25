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

package org.apache.nlpcraft.server.ctxword

import java.util

import com.typesafe.scalalogging.LazyLogging

import scala.collection.JavaConverters._
import scala.collection._

case class NCContextWordFactors(
    private val meta: Map[String, AnyRef],
    private val elementsIds: Set[String],
    private val defaults: Map[String, Double]
) extends LazyLogging {
    private val m: Map[String, Map[String, Double]] = initialize()

    private def initialize(): Map[String, Map[String, Double]] = {
        val allDefs = elementsIds.map(id ⇒ id → defaults).toMap

        meta.get("ctx.words.factors") match {
            case Some(v) ⇒
                v.asInstanceOf[util.HashMap[String, util.Map[String, Double]]].asScala.
                    map(p ⇒ p._1 → p._2.asScala).
                        flatMap { case (elemId, elemMeta) ⇒
                        allDefs.get(elemId) match {
                            case Some(elemDflts) ⇒
                                val unsexp = elemDflts.keySet -- elemMeta.keySet

                                if (unsexp.nonEmpty) {
                                    println("elemData="+elemMeta.keySet)
                                    println("elemDefaults="+elemDflts.keySet)

                                    logger.warn(s"Unexpected factors: {$unsexp} of element ID: '$elemId'.")
                                }

                                Some(elemId → (elemDflts ++ elemMeta))
                            case None ⇒
                                logger.warn(s"Unexpected element ID: '$elemId' data skipped.")

                                None
                        }
                    }.toMap

            case None ⇒ Map.empty ++ allDefs
        }
    }

    private val mins: Map[String, Double] = defaults.keySet.map(name ⇒ name → m.map(p ⇒ p._2(name)).min).toMap

    def get(elemId: String, param: String): Double = m(elemId)(param)
    def getMin(param: String): Double = mins(param)
}
