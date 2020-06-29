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

package org.apache.nlpcraft.server.mdo

import org.apache.nlpcraft.server.mdo.impl._

import scala.collection.{Map, Set}

/**
  *
  */
@NCMdoEntity(sql = false)
case class NCExampleMdo(@NCMdoField words: Seq[String], @NCMdoField substitutions: Map[Int/*Position*/, String/*POS*/])

/**
  *
  */
@NCMdoEntity(sql = false)
case class NCContextWordFactors(
    @NCMdoField elementsFactors: Map[String, Map[String, Double]],
    @NCMdoField minimalFactors: Map[String, Double]
) {
    def get(elemId: String, param: String): Double = elementsFactors(elemId)(param)
    def getMin(param: String): Double = minimalFactors(param)
}

/**
  * Probe model context word config MDO.
  */
@NCMdoEntity(sql = false)
case class NCContextWordConfigMdo(
    @NCMdoField synonyms: Map[String /*Element ID*/, Map[String /*Synonym stem*/, String /*Value*/]],
    @NCMdoField contextWords: Map[String /*Element ID*/, Set[String]/*Synonym texts*/],
    @NCMdoField examples: Map[String /*Element ID*/, Seq[NCExampleMdo]/*Examples*/],
    @NCMdoField poses: Set[String],/*All possible POSes of context words*/
    @NCMdoField factors: NCContextWordFactors
) {
    require(synonyms.size == contextWords.size)
    require(synonyms.size == examples.size)
}