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

package org.nlpcraft.probe.mgrs.nlp.enrichers.suspicious

import java.io.Serializable

import io.opencensus.trace.Span
import org.nlpcraft.common.{NCE, NCService}
import org.nlpcraft.common.nlp._
import org.nlpcraft.probe.mgrs.NCModelDecorator
import org.nlpcraft.probe.mgrs.nlp.NCProbeEnricher

import scala.collection.Map

/**
  * Suspicious words enricher.
  */
object NCSuspiciousNounsEnricher extends NCProbeEnricher {
    override def start(parent: Span = null): NCService = startScopedSpan("start", parent) { _ ⇒
        super.start()
    }

    override def stop(parent: Span = null): Unit = startScopedSpan("stop", parent) { _ ⇒
        super.stop()
    }

    @throws[NCE]
    override def enrich(mdl: NCModelDecorator, ns: NCNlpSentence, senMeta: Map[String, Serializable], parent: Span = null): Boolean =
        startScopedSpan("enrich", parent,
            "srvReqId" → ns.srvReqId,
            "modelId" → mdl.model.getId,
            "txt" → ns.text) { _ ⇒
            val suspToks = ns.filter(t ⇒ mdl.suspiciousWordsStems.contains(t.stem))

            if (suspToks.nonEmpty) {
                suspToks.foreach(_.getNlpNote += "suspNoun" → true)

                true
            }
            else
                false
        }
}
