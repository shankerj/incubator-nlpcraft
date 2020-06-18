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

/**
  * All parameters.
  */
object NCContextWordParameter {
    private final val DFLT_LIMIT = 20
    private final val DFLT_TOTAL_SCORE = 0
    private final val DFLT_FTEXT_SCORE = 0.25
    private final val DFLT_BERT_SCORE = 0

    // Configuration parameters.
    // Configuration request limit for each processed example.
    final val CONF_LIMIT = 1000
    // Minimal score for requested words for each processed example.
    final val CONF_MIN_SCORE = 1
    // If we have a lot of context words candidates, we choose top 50%.
    final val CONF_TOP_FACTOR = 0.5
    // If we have small context words candidates count, we choose at least 3.
    final val CONF_TOP_MIN = 3

    // Enricher parameters.
    // Configuration when we try to find context words for words nouns using initial sentence.
    final val MIN_SENTENCE_SCORE = 0.5
    final val MIN_SENTENCE_FTEXT = 0.5

    // Configuration when we try to find context words for words nouns using substituted examples.
    final val MIN_EXAMPLE_SCORE = 0.8
    // Context words for all examples should satisfy it (not so strong)
    final val MIN_EXAMPLE_ALL_FTEXT = 0.2
    // At least on context word with this score must be found.
    final val MIN_EXAMPLE_BEST_FTEXT = 0.5
}

/**
  * Default ContextWord server parameters.
  *
  * @param limit
  * @param totalScore
  * @param ftextScore
  * @param bertScore
  */
case class NCContextWordParameter(
    limit: Int = NCContextWordParameter.DFLT_LIMIT,
    totalScore: Double = NCContextWordParameter.DFLT_TOTAL_SCORE,
    ftextScore: Double = NCContextWordParameter.DFLT_FTEXT_SCORE,
    bertScore: Double = NCContextWordParameter.DFLT_BERT_SCORE
)