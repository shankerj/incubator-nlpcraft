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
 *
 */

package org.apache.nlpcraft.probe.mgrs.nlp.enrichers.function

import java.io
import _root_.io.opencensus.trace.Span
import org.apache.nlpcraft.common.nlp.{NCNlpSentence, NCNlpSentenceNote, NCNlpSentenceToken}
import org.apache.nlpcraft.probe.mgrs.NCModelDecorator
import org.apache.nlpcraft.probe.mgrs.nlp.NCProbeEnricher
import org.apache.nlpcraft.probe.mgrs.nlp.enrichers.limit.NCLimitEnricher.{isUserNotValue, startScopedSpan, techWords}
import scala.collection.{Seq, mutable}

object NCAverageEnricher extends NCFunctionEnricher {
    override def funcType: String = "average"

    case class Match(matched: Seq[NCNlpSentenceToken],
                     matchedHead: NCNlpSentenceToken,
                     refNotes: Set[String],
                     refIndexes: java.util.List[Int]
                    )

    /**
     *
     * Processes this NLP sentence.
     *
     * @param mdl     Model decorator.
     * @param ns      NLP sentence to enrich.
     * @param senMeta Sentence metadata.
     * @param parent  Span parent.
     */
    override def enrich(mdl: NCModelDecorator,
                        ns: NCNlpSentence,
                        senMeta: collection.Map[String, io.Serializable],
                        parent: Span): Unit = startScopedSpan("enrich", parent,
        "srvReqId" → ns.srvReqId,
        "modelId" → mdl.model.getId,
        "txt" → ns.text) { _ ⇒
        val notes = mutable.HashSet.empty[NCNlpSentenceNote]

        for (toks ← ns.tokenMixWithStopWords() if validImportant(ns, toks))
            tryToMatch(toks) match {
                case Some(m) ⇒
                    for (refNote ← m.refNotes) {
                        val note = NCNlpSentenceNote(
                            Seq(m.matchedHead.index),
                            TOK_ID,
                            "type" → funcType,
                            "indexes" → m.refIndexes,
                            "note" → refNote
                        )

                        if (!notes.exists(n ⇒ ns.notesEqualOrSimilar(n, note))) {
                            notes += note

                            m.matched.filter(_ != m.matchedHead).foreach(_.addStopReason(note))

                            m.matchedHead.add(note)
                        }
                    }
                case None ⇒ // No-op.
            }
    }


    private def validImportant(ns: NCNlpSentence, toks: Seq[NCNlpSentenceToken]): Boolean = ???

    private def tryToMatch(toks: Seq[NCNlpSentenceToken]): Option[Match] = ???
}
